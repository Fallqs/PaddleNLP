import numpy as np
import json
from paddlenlp.transformers import GPTTokenizer
import time
import multiprocessing
import argparse
from io import StringIO
import time
import random


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='data input/output')
    group.add_argument('--seq_len',
                       type=int,
                       default=1024,
                       required=False,
                       help='Path to input JSON files.')
    group.add_argument('-i', '--input_path',
                       type=str,
                       required=True,
                       help='Path to input JSON files.')
    group.add_argument('-o', '--output_prefix',
                       type=str,
                       required=True,
                       help='Output prefix to store output file.')
    group.add_argument(
        '--data_format',
        type=str,
        default='text',
        choices=['JSON'],
        help='Only support json format for now. One document per line.')
    group.add_argument(
        '--json_key',
        type=str,
        default='text',
        help='For JSON format. Space separate listed of keys to extract from json')
    group = parser.add_argument_group(title='common config')
    group.add_argument('--append_eos',
                       action='store_true',
                       help='Append an <eos> token to the end of a document.')
    group.add_argument('--log_interval',
                       type=int,
                       default=100,
                       help='Interval between progress updates')
    group.add_argument('--workers',
                       type=int,
                       default=1,
                       help='Number of worker processes to launch')
    group.add_argument("--vocab_file",
                       type=str,
                       default='./data_tools/code-vocab.json',
                       help="Path to the vocab file")
    group.add_argument("--merge_file",
                       type=str,
                       default='./data_tools/code-merges.txt',
                       help="Path to the BPE merge file (if necessary).")
    return parser.parse_args()


class _Sample(object):
    def __init__(self, tokenizer, head: list, comma: int,
                 seg_id: int = 0, note_id: int = 50257, size: int = 1025):
        self.tk, self.seg_id, self.note_id, self.size = tokenizer, seg_id, note_id, size
        self.head, self.body, self.comma = head, [], comma
        self._head = head[:]
        self.l = len(self.head) + 1
        self.mp = dict()

    def add_line(self, cont: str, vids: list):
        cont = self.tk(cont)
        j = 0
        for i, v in enumerate(cont):
            if v == self.note_id:
                assert j < len(vids), f"{cont}"
                cont[i], j = cont[i] + vids[j], j + 1
        self.body.extend(cont)
        self.l += len(cont)
        return self.l >= self.size

    def add_var(self, name: str, voft: int):
        cont = self.tk(name + ':')
        self.mp[self.note_id + voft] = self.tk(name)
        # print(name, cont, self.note_id, self.comma)
        self.head.extend(cont + [self.note_id + voft, self.comma])
        self.l += len(cont) + 2
        return self.l >= self.size

    def collect(self):
        self.head.append(self.seg_id)
        msk = len(self.head)
        self.head += self.body
        if len(self.head) >= self.size:
            return self.head[:self.size], msk
        return self.head, msk

    def wasted(self):
        return len(self.head) >= self.size >> 1

    def clear(self):
        self.head, self.body = self._head[:], []
        self.l = len(self.head) + 1
        self.mp = dict()

    def decode(self, ids: list[int]):
        ret = []
        for i in ids:
            if i < self.note_id or i not in self.mp:
                ret.append(i)
            else:
                ret.extend(self.mp[i])
        # print(ret)
        return ret

    def __len__(self):
        return self.l


class Sampler(object):
    lang = {'False', 'await', 'else', 'import', 'pass',
            'None', 'break', 'except', 'in', 'raise',
            'True', 'class', 'finally', 'is', 'return',
            'and', 'continue', 'for', 'lambda', 'try',
            'as', 'def', 'from', 'nonlocal', 'while',
            'assert', 'del', 'global', 'not', 'with',
            'async', 'elif', 'if', 'or', 'yield'}
    builtin = {
        'set', 'list', 'dict', 'bool', 'str', 'chr',
        'ord', 'int', 'float', 'format', 'map',
        'filter', 'sum', 'max', 'min', 'mean', 'open',
        'enumerate', 'zip', 'range', 'print', 'input',
        'split', 'self', 'append', 'extend', 'join',
        'pop', 'object', 'match', 'case'
    }

    charset = set([chr(ord('a') + i) for i in range(26)] +
                  [chr(ord('A') + i) for i in range(26)] +
                  ['_'])
    numset = set('.eE-')
    digiset = set('0123456789')
    quo = {"'": 1, '"': 2, "'''": 3, '"""': 4}
    iquo = ['', "'", '"', "'''", '"""']
    ignore = lang | builtin | charset

    nan, idt, num, ostr, anno = 0, 1, 2, 3, 4
    var, func = -1, -2

    IDT = 'cVf'

    def __init__(self, cont, tokenizer, seg_id: int = 0, seq_length: int = 1025,
                 filter_rate: int = 0.5):
        # sp: item: [name: str, begin: int, end: int, type: int]
        self.cont, self.sp = '\n' + cont + '\n\n', []
        self.stat = self.nan
        # mp: {identifier: str -> [rank: int, first_appearance: int]}
        self.mp = dict()
        self.tokenizer, self.seg_id, self.seq_length = tokenizer, seg_id, seq_length
        self.head, self.comma = tokenizer('var = '), tokenizer(',')[0]
        self.err = False
        self.filter_rate = filter_rate
        try:
            self._load()
        except IndexError:
            self.err = True

    @staticmethod
    def encoding_tokenizer(tokenizer, seq_length: int = 1025, idt: str or None = None):
        if idt is not None:
            Sampler.IDT = idt
        size = tokenizer.add_tokens([f'<{Sampler.IDT}{i}>' for i in range(seq_length >> 2)], True)
        assert size == seq_length >> 2, 'Unaccepted IDT'
        return tokenizer

    @staticmethod
    def decoding_tokenizer(tokenizer, seq_length: int = 1025, idt: str or None = None):
        if idt is None:
            idt = Sampler.IDT
        size = tokenizer.add_tokens([idt + str(i) for i in range(seq_length >> 2)], False)
        assert size == seq_length >> 2, 'Unaccepted IDT'
        return tokenizer

    @staticmethod
    def builtin(idt: str) -> bool:
        return len(idt) > 4 and idt.startswith('__') and idt.endswith('__')

    def _load(self):
        buf, isf, onf, bkslash, scnt, ecnt, styp = '', False, False, False, 0, 0, ''
        fbrc, bcnt = [(-1, '')], 0

        def _nan(i, x):
            nonlocal buf, onf, scnt, ecnt, styp, bcnt, fbrc
            if x in self.charset:
                self.stat = self.idt
                buf = x
            elif x in self.digiset:
                self.stat = self.num
            elif x in self.quo:
                q = self.cont[i:i + 3]
                styp = q if q[1] == x and q[2] == x else x
                onf, scnt = isf, int(
                    q.startswith("'''") or q.startswith('"""')) << 1
                self.stat = self.ostr
            elif x == '#':
                self.stat = self.anno
            elif x == '}' and bcnt == fbrc[-1][0]:
                styp, onf = fbrc[-1][1], True
                self.stat = self.ostr
                fbrc.pop()
                if not fbrc:
                    raise IndexError

        def _idt(i, x):
            nonlocal buf
            if x in self.charset | self.digiset:
                buf += x
            else:
                self.stat = self.nan
                if x not in self.quo:
                    # left-closed & right-open interval
                    self.sp.append([buf, i - len(buf), i, self.idt])
                buf = ''
                _nan(i, x)

        def _num(i, x):
            if x not in self.numset | self.digiset:
                self.stat = self.nan
                _nan(i, x)

        def _ostr(i, x):
            nonlocal scnt, ecnt, styp
            if scnt:
                scnt -= 1
            elif ecnt:
                ecnt -= 1
                if not ecnt:
                    self.stat = self.nan
            elif onf and x == '{':
                fbrc.append((bcnt + 1, styp))
                self.stat = self.nan
            elif not bkslash and self.cont[i:i + 3].startswith(styp):
                if len(styp) == 1:
                    self.stat = self.nan
                else:
                    ecnt = 2

        def _anno(i, x):
            if x == '\n':
                self.stat = self.nan

        jmp = [_nan, _idt, _num, _ostr, _anno]

        for i, x in enumerate(self.cont):
            jmp[self.stat](i, x)
            isf, bkslash = x == 'f', x == '\\' and not bkslash
            if self.stat != self.anno:
                bcnt += int(x == '{') - int(x == '}')
        if buf and self.stat == self.idt:
            self.sp.append([buf, len(self.cont) - len(buf),
                            len(self.cont), self.idt])
        self.sp.append(
            ['', len(self.cont) + 10, len(self.cont) + 10, self.idt])

    def _filter(self, encoding: bool = True):
        def chk(x) -> bool:
            t = x[0]
            return t not in self.ignore and not self.builtin(t)

        self.sp = list(filter(chk, self.sp))
        vids, vidx = [u[0] for u in self.sp], [0] + [len(u) + 1 for u in self.cont.split('\n')]
        j, pre = 0, 0
        for i, ix in enumerate(vidx):
            pre = ix = ix + pre
            while self.sp[j][1] < ix:
                it = self.sp[j]
                if it[0] not in self.mp:
                    self.mp[it[0]] = i
                j += 1
            vidx[i] = j

        buf, lst, idt0 = StringIO(), 0, f'<{self.IDT}0>' if encoding else self.IDT + '0'
        self.sp.pop()
        for it in self.sp:
            buf.write(self.cont[lst:it[1]])
            buf.write(idt0)
            lst = it[2]
        buf.write(self.cont[lst:])
        buf.seek(0)
        cont = buf.readlines()
        buf.close()
        return vids, vidx, cont

    def prompt(self, siz: int):
        vids, vidx, cont = self._filter(encoding=False)
        j, vmp = 0, dict()
        cur = _Sample(self.tokenizer, self.head, self.comma,
                      self.seg_id, self.tokenizer(f'{self.IDT}0')[0],
                      siz)

        for i, line in enumerate(cont[:-1]):
            vl = vids[vidx[i]:vidx[i + 1]]
            for name in vl:
                if name not in vmp:
                    cur.add_var(name, j)
                    vmp[name], j = j, j + 1
            vl = list(map(lambda v: vmp[v], vl))
            if cur.add_line(line, vl):
                break
        ids, _ = cur.collect()
        return ids[:-1], cur.decode  #, ''.join(cont)

    def collect(self):
        ret = ([], [], [])  # item = ([ids], [i: loss_mask_head], [j: len])
        if self.err:
            return ret
        vids, vidx, cont = self._filter()

        start, j, vmp = 0, 0, dict()
        cur = _Sample(self.tokenizer, self.head, self.comma,
                      self.seg_id, self.tokenizer(f'<{self.IDT}0>')[0],
                      self.seq_length)

        def export(i=0):
            nonlocal start, j, cur, vmp
            # print(cur.head, cur.body)
            if not cur.wasted():
                ids, msk = cur.collect()
                ret[0].extend(ids)
                ret[1].append(len(ids))
                ret[2].append(msk)
            cur.clear()
            start, j, vmp = i + 1, 0, dict()

        for i, line in enumerate(cont):
            vl = vids[vidx[i]:vidx[i + 1]]
            for name in vl:
                if name not in vmp:
                    if self.mp[name] < start or random.random() < self.filter_rate:
                        cur.add_var(name, j)
                    vmp[name], j = j, j + 1
            vl = list(map(lambda v: vmp[v], vl))
            if cur.add_line(line, vl):
                export(i)

        export()
        return ret


def process(jsonl, key: str, tokenizer, seq_length: int = 1024):
    def tk(x):
        return tokenizer(x)['input_ids']

    return Sampler(json.loads(jsonl)[key], tk,
                   tokenizer.eos_token_id, seq_length + 1).collect()


def prompt_ids(content: str, tokenizer, size: int = 256):
    """
    For example:
    from tio_gen import prompt_ids

    args = get_args()
    tokenizer = GPTTokenizer(args.vocab_file, args.merge_file)
    ids = prompt_ids(args.content, tokenizer, args.size)

    print(tokenizer.convert_ids_to_string(ids))
    """

    def tk(x):
        return tokenizer(x)['input_ids']

    return Sampler(content, tk, tokenizer.eos_token_id, size).prompt(size)


def test(input: str = 'code_python'):
    t = time.perf_counter()
    ids = np.load(input + '_ids_tio.npy')
    npz = np.load(input + '_idx_tio.npz')
    idx, los = npz['idx'], npz['los']
    print('loaded:', time.perf_counter() - t)

    print('los:', np.mean(los), max(los))
    print('len:', idx[-1] / len(los))

    for i, p in enumerate(los):
        sample = ids[idx[i]:idx[i + 1]]
        # print(eos, sample[p-3:p+3])
        assert sample[p - 1] == 0

    return ids, idx, los


def test2(jl, tke, tkd):
    def tk(x):
        return tke(x)['input_ids']

    for i in range(3):
        ids, dec, cont = Sampler(json.loads(jl.readline())['text'], tk, tke.eos_token_id).prompt(1025)
        print('SOURCE:')
        print(cont)
        print('---------------------------\nTOKENIZE:')
        print(tk(cont))
        print(tkd.encode(cont))
        print(tkd.tokenize(cont))
        print('---------------------------\nDECODE:')
        print(tkd.decode(tk(cont), spaces_between_special_tokens=False))
        print('===========================')


if __name__ == '__main__':
    args = get_args()
    seq_len = args.seq_len
    tk = GPTTokenizer(args.vocab_file, args.merge_file)
    tk = Sampler.encoding_tokenizer(tk, seq_len + 1)
    # tkd = Sampler.decoding_tokenizer(GPTTokenizer(args.vocab_file, args.merge_file, seq_len + 1))
    print(len(tk))
    eos = tk.eos_token_id

    # ids, idx, los = test('finetune10/code_python')
    # for i in range(len(idx) - 1):
    #     print(tk.decode(ids[idx[i]:idx[i + 1]]))
    #     print('---------------------------------')

    jl = open(args.input_path, 'r', encoding='utf-8')

    # test2(jl, tk, tk)

    # jl.close()

    t = time.perf_counter()
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
        ret = p.starmap(process, map(lambda x: (x, 'text', tk, seq_len), jl))
    print('generated, duration =', time.perf_counter() - t)
    jl.close()

    t = time.perf_counter()
    ids, idx, los = [], [0], []
    for s, x, l in ret:
        ids.extend(s)
        idx.extend(x)
        los.extend(l)
    ids = np.array(ids, dtype='int32')
    idx = np.cumsum(np.array(idx, dtype='int64'))
    los = np.array(los)
    with open(args.output_prefix + '_ids_tio.npy', 'wb') as f:
        np.save(f, ids, allow_pickle=True)
    with open(args.output_prefix + '_idx_tio.npz', 'wb') as f:
        np.savez(f, idx=idx, los=los)
    print('saved, duration =', time.perf_counter() - t)
