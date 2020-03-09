from collections import defaultdict
from scipy.stats import entropy
import argparse

def setup_data(path):
    print("[loading fbdialog data:" + path + "]")
    with open(path) as read:
        start = True
        x = ''
        reward = 0
        last_conv_id = None
        cnt = -1
        for line in read:
            cnt+=1
            line = line.strip().replace('\\n', '\n')
            if len(line) == 0:
                # empty response
                continue

            # first, get conversation index -- '1' means start of episode
            space_idx = line.find(' ')
            if space_idx == -1:
                # empty line, both individuals are saying whitespace
                conv_id = int(line)
            else:
                conv_id = int(line[:space_idx])

            # split line into constituent parts, if available:
            # x<tab>y<tab>reward<tab>label_candidates
            # where y, reward, and label_candidates are optional
            split = line[space_idx + 1 :].split('\t')

            # remove empty items and strip each one
            for i in range(len(split)):
                word = split[i].strip()
                if len(word) == 0:
                    split[i] = ''
                else:
                    split[i] = word
            # Empty reward string same as None
            if len(split) > 2 and split[2] == '':
                split[2] = None

            # now check if we're at a new episode
            if last_conv_id is None or conv_id <= last_conv_id:
                x = x.strip()
                if x:
                    yield cnt,[x, None, reward], start
                start = True
                reward = 0
                # start a new episode
                x = split[0]
            else:
                if x:
                    # otherwise add current x to what we have so far
                    x = '{x}\n{next_x}'.format(x=x, next_x=split[0])
                else:
                    x = split[0]
            last_conv_id = conv_id
            if len(split) > 2 and split[2]:
                reward += float(split[2])

            if len(split) > 1 and split[1]:
                # only generate an example if we have a y
                split[0] = x
                # split labels
                split[1] = split[1].split('|')
                if len(split) > 3:
                    # split label_candidates
                    split[3] = split[3].split('|')
                if len(split) > 2:
                    split[2] = reward
                else:
                    split.append(reward)
                if start:
                    yield cnt, split, True
                    start = False
                else:
                    yield cnt, split, False
                # reset x in case there is unlabeled data still left
                x = ''
                reward = 0
        if x:
            yield cnt, [x, None, reward], start


def get_st_pairs(path):
    for (cnt, (x, y, _), _) in setup_data(path):
        if not y: continue
        yield cnt, x, y[0]


def get_word_frequencies(train_exs):
    word_freqs = defaultdict(int)
    for (x, y) in train_exs:
        for utt in [x, y]:
            for w in utt.split():
                word_freqs[w] += 1
    return word_freqs


def get_cond_maps(train_exs):
    p_ts = defaultdict(lambda: [0, defaultdict(int)])
    p_st = defaultdict(lambda: [0, defaultdict(int)])
    for (_, s, t) in train_exs:
        p_ts[s][0] += 1
        p_st[t][0] += 1
        p_ts[s][1][t] += 1
        p_st[t][1][s] += 1
    return p_ts, p_st


def compute_h(cond_dict):
    return {source: entropy(list(cond_dict[source][1].values())) for source in cond_dict.keys()}


def filter_dataset(path, out_path, threshold=1.5):
    train_exs = get_st_pairs(path)
    p_ts, p_st = get_cond_maps(train_exs)
    h_tgt = compute_h(p_ts)
    h_src = compute_h(p_st)
    skipped = 0
    with open(path) as f:
        with open(out_path, 'w') as g:
            line_cnt = -1
            for cnt, s, t in get_st_pairs(path):
                while line_cnt < cnt:
                    train_line = f.readline()
                    line_cnt += 1
                if h_tgt[s] <= threshold and h_src[t] <= threshold:
                    g.write(train_line)
                else:
                    #                     print(s, ' => ' , t)
                    # print(train_line, ' => ', (t if h_src[t] > threshold else s), ' triggered!')
                    skipped += 1
    print(f'skipped {skipped} out of {cnt+1} lines ({skipped / (cnt+1) * 100:<.3f}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--threshold', default=1.5)
    opt = parser.parse_args()
    filter_dataset(opt.train_path, opt.out_path,threshold=opt.threshold)
