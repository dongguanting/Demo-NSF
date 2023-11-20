import nlpaug.augmenter.char as nac
import random
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.word as naw
from sentence_transformers import SentenceTransformer, util


test_file_path = "xxx/multi-noise/test"
train_file_path = "xxx/multi-noise/train"
valid_file_path = "/home/htf2022/t5_scene/data/datasets/multi-noise/valid"

train_dict = {"train": (train_file_path + "/seq.in", train_file_path + "/seq.out")}
train_ner_dict = {"train": train_file_path + "/train.txt"}

valid_dict = {"valid": (valid_file_path + "/seq.in", valid_file_path + "/seq.out")}
valid_ner_dict = {"valid": valid_file_path + "/valid.txt"}

test_dict = {
    "appendirr": (
        test_file_path + "/AppendIrr/seq.in",
        test_file_path + "/AppendIrr/seq.out",
    ),
    "append+sub": (
        test_file_path + "/appendirr+sub/seq.in",
        test_file_path + "/appendirr+sub/seq.out",
    ),
    "clean": (
        test_file_path + "/clean_test/seq.in",
        test_file_path + "/clean_test/seq.out",
    ),
    "enttypos": (
        test_file_path + "/EntTypos/seq.in",
        test_file_path + "/EntTypos/seq.out",
    ),
    "sub": (
        test_file_path + "/sub/seq.in",
        test_file_path + "/sub/seq.out",
    ),
    "typos+append": (
        test_file_path + "/typos+appendirr/seq.in",
        test_file_path + "/typos+appendirr/seq.out",
    ),
    "typos+sub": (
        test_file_path + "/typos+sub/seq.in",
        test_file_path + "/typos+sub/seq.out",
    ),
    "typos+sub+append": (
        test_file_path + "/typos+sub+appendirr/seq.in",
        test_file_path + "/typos+sub+appendirr/seq.out",
    ),
}
test_ner_dict = {
    "appendirr": test_file_path + "/AppendIrr/test.txt",
    "append+sub": test_file_path + "/appendirr+sub/test.txt",
    "clean": test_file_path + "/clean_test/test.txt",
    "enttypos": test_file_path + "/EntTypos/test.txt",
    "sub": test_file_path + "/sub/test.txt",
    "typos+append": test_file_path + "/typos+appendirr/test.txt",
    "typos+sub": test_file_path + "/typos+sub/test.txt",
    "typos+sub+append": test_file_path + "/typos+sub+appendirr/test.txt",
}

char_dict = {"char": (train_file_path + "/char/seq.in", train_file_path + "/char/seq.out")}

def convert_to_ner_data(data_path_dict, out_path_dict):
    """
    将各个数据集转换为NER形式
    """
    for k, v in data_path_dict.items():
        with open(v[0], "r") as f1, open(v[1], "r") as f2, open(
            out_path_dict[k], "w"
        ) as f3:
            for ln1, ln2 in zip(f1, f2):
                ln1 = ln1.strip().split()
                ln2 = ln2.strip().split()
                for ch1, ch2 in zip(ln1, ln2):
                    f3.write("{} {}".format(ch1, ch2))
                    f3.write("\n")
                f3.write("\n")

def char_aug(train_dict, char_dict):
    aug = nac.KeyboardAug(aug_char_p=0.2, aug_word_p=0.2, include_special_char=False)
    f = open(train_dict["train"][0], 'r')
    l = open(train_dict["train"][1], 'r')
    f_out = open(char_dict["char"][0], 'w')
    l_out = open(char_dict["char"][1], 'w')
    for ln in f:
        ln = ln.strip()
        ln = aug.augment(ln)
        f_out.write("".join(ln))
        f_out.write("\n")
    for ln in l:
        l_out.write(ln.strip())
        l_out.write("\n")
        
mask_out_dict = {"train": (train_file_path + "/mask.in", train_file_path + "/mask.out")}
        
def convert_to_mask_data(data_path_dict, mask_out_dict):
    for k, v in data_path_dict.items():
        if k not in mask_out_dict:
            continue
        with open(v[0], 'r') as f1, open(mask_out_dict[k][0], 'w') as f2, open(mask_out_dict[k][1], 'w') as f3:
            for ln in f1:
                ln = ln.strip().split()
                ln_in = ln.copy()
                ln_out = ln.copy()
                mask_idx = random.randint(0, len(ln) - 1)
                if mask_idx == 0:
                    ln_in[mask_idx] = "<extra_id_0>"
                    ln_out = ln[mask_idx] + " <extra_id_0>"
                    ln_in = " ".join(ln_in)
                    f2.write(ln_in)
                    f2.write('\n')
                    f3.write(ln_out)
                    f3.write('\n')
                elif mask_idx == len(ln) - 1:
                    ln_in[mask_idx] = "<extra_id_0>"
                    ln_out = "<extra_id_0> " + ln[mask_idx]
                    ln_in = " ".join(ln_in)
                    f2.write(ln_in)
                    f2.write('\n')
                    f3.write(ln_out)
                    f3.write('\n')
                else:
                    ln_in[mask_idx] = "<extra_id_0>"
                    ln_out = "<extra_id_0> " + ln[mask_idx] + " <extra_id_1>"
                    ln_in = " ".join(ln_in)
                    f2.write(ln_in)
                    f2.write('\n')
                    f3.write(ln_out)
                    f3.write('\n')
        
word_dict = {"word": (train_file_path + "/word/seq1.in", train_file_path + "/word/seq.out")}        
            
def word_aug(train_dict, char_dict):
    aug = naw.RandomWordAug()
    f = open(train_dict["train"][0], 'r')
    f_out = open(char_dict["word"][0], 'w')
    for ln in f:
        ln = ln.strip()
        ln = aug.augment(ln)
        f_out.write("".join(ln))
        f_out.write("\n")
        
def convert_to_demonstration_data(all_data_dict, train_demons_dict, train_demons_out, num=2):
    '''
        构造demonstration数据，根据sbert召回
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for k, v in all_data_dict.items():
        lines = []
        f = open(v[0], 'r')
        ff = open(v[1], 'r')
        lines = f.readlines()
        lines_l = ff.readlines()
        assert len(lines) == len(lines_l)
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines_l[i] = lines_l[i].replace('\n', '')
        demons = []
        labels = []
        for k1, v1 in train_demons_dict.items():
            if k1 != k:
                continue
            f1 = open(train_demons_dict[k1][0], 'r')
            f2 = open(train_demons_out[k1][0], 'w')
            f2_out = open(train_demons_out[k1][1], 'w')
            embeddings = model.encode(lines)
            idx = 0
            eps = 1e-4
            for i, ln in enumerate(f1):
                # if i == 2:
                #     pdb.set_trace()
                emb = model.encode(ln)
                cos_sim = util.cos_sim(emb, embeddings)
                cos_sim_sorted = ((-cos_sim).argsort().numpy().tolist())[0]
                for i in range(len(cos_sim_sorted)):
                    if abs(cos_sim[0][cos_sim_sorted[i]].item() - 1.0) > eps:
                        idx = i
                        break
                cos_sim_sorted = cos_sim_sorted[idx: (num + idx)]
                for id in cos_sim_sorted:
                    demons.append(" ".join(lines[id].split()))
                    labels.append(" ".join(lines_l[id].split()))
                f2.write(' ||| '.join(demons))
                f2.write('\n')
                f2_out.write(' ||| '.join(labels))
                f2_out.write('\n')
                demons = []
                labels = []

train_demons_out = {
    "train": (train_file_path + "/train_demons.in", train_file_path + "/train_demons.out")}
test_demons_out = {
    "appendirr": (
        test_file_path + "/AppendIrr/demons.in",
        test_file_path + "/AppendIrr/demons.out",
    ),
    "append+sub": (
        test_file_path + "/appendirr+sub/demons.in",
        test_file_path + "/appendirr+sub/demons.out",
    ),
    "clean": (
        test_file_path + "/clean_test/demons.in",
        test_file_path + "/clean_test/demons.out",
    ),
    "enttypos": (
        test_file_path + "/EntTypos/demons.in",
        test_file_path + "/EntTypos/demons.out",
    ),
    "sub": (
        test_file_path + "/sub/demons.in",
        test_file_path + "/sub/demons.out",
    ),
    "typos+append": (
        test_file_path + "/typos+appendirr/demons.in",
        test_file_path + "/typos+appendirr/demons.out",
    ),
    "typos+sub": (
        test_file_path + "/typos+sub/demons.in",
        test_file_path + "/typos+sub/demons.out",
    ),
    "typos+sub+append": (
        test_file_path + "/typos+sub+appendirr/demons.in",
        test_file_path + "/typos+sub+appendirr/demons.out",
    ),
}
valid_demons_out = {
    "valid": (valid_file_path + "/valid_demons.in", valid_file_path + "/valid_demons.out")}

def convert_to_classify_data():
    f1 = open(train_file_path + "/seq.in", 'r')
    f2 = open(train_file_path + "/char/seq.in", 'r')
    f3 = open(train_file_path + "/word/seq.in", 'r')
    f4 = open(train_file_path + "/word/seq1.in", 'r')
    f_out = open(train_file_path + "/classify.in", 'a')
    f_out_1 = open(train_file_path + "/classify.out", 'a')
    for i, ln in enumerate(f1):
        if i < 3271:
            ln = ln.strip()
            f_out.write(ln)
            f_out.write("\n")
            f_out_1.write("This is a clean sentence.")
            f_out_1.write("\n")
        else:
            break
    for i, ln in enumerate(f2):
        if i < 3271:
            ln = ln.strip()
            f_out.write(ln)
            f_out.write("\n")
            f_out_1.write("This is a typos sentence.")
            f_out_1.write("\n")
        else:
            break
    for i, ln in enumerate(f3):
        if i < 3271:
            ln = ln.strip()
            f_out.write(ln)
            f_out.write("\n")
            f_out_1.write("This is a speech sentence.")
            f_out_1.write("\n")
        else:
            break
    for i, ln in enumerate(f4):
        if i < 3271:
            ln = ln.strip()
            f_out.write(ln)
            f_out.write("\n")
            f_out_1.write("This is a simplify sentence.")
            f_out_1.write("\n")
        else:
            break