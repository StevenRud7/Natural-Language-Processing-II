import numpy as np
import scipy.linalg as la
from scipy.linalg import norm
from scipy.spatial.distance import cosine, euclidean
import random
import tarfile
import os
import pandas as pd

#PART 1!!!

with open('dist_sim_data.txt') as f:
    corpus = f.readlines()

# create features
fd = dict()
for sentence in corpus:
    for word in sentence.split():
        if word in fd:
            fd[word] =fd[word] +1
        else:
            fd[word]= 1
        
# create co-occurrence matrix
C = np.zeros((len(fd), len(fd)),dtype = int)

f_list = list(fd.keys())


for w, w_feat in enumerate(f_list):
    for c, c_feat in enumerate(f_list):
        if w == c:
            C[w, c] = fd[w_feat] #or fd[f_list[w]]
        else:
            for sent in corpus:
                
                sent = sent.split()
                #sent = " ".join(sent)
                # C[f_list.index(w_feat), f_list.index(sent[l2])] += 1
                w_c = f_list[w] + ' ' + f_list[c] #try doing w_feat and c_feat instead
                c_w = f_list[c] + ' ' + f_list[w]
                for x in range(len(sent)-1):
                    if (sent[x] + " " + sent[x+1]) == w_c:
                        C[w,c] +=1
                    if (sent[x] + " " + sent[x+1]) == c_w:
                        C[w,c] +=1

print(C)

C = (C * 10) + 1

print(C)

# Find P(w)
P_w = np.sum(C, axis=1) / np.sum(C)

# Find P(c)
P_c = np.sum(C, axis=0) / np.sum(C)

# Find P(w, c)
P_wc = C / np.sum(C)

# Find PPMI
PPMI = np.zeros(C.shape)

for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        if P_wc[i][j] == 0:
            PPMI[i][j] = 0
        else:
            PPMI[i][j] = max(np.log2(P_wc[i][j]/(P_w[i]*P_c[j])), 0)

print("PPMI is \n", PPMI)


# use PPMI matrix as weighted count matrix
weighted_C = PPMI

# compare the word vector for "dogs" before and after PPMI reweighting
dogs_vector_before = C[f_list.index('dogs')]
dogs_vector_after = weighted_C[f_list.index('dogs')]

print("Dogs vector before PPMI reweighting:", dogs_vector_before)
print("Dogs vector after PPMI reweighting:", dogs_vector_after)




# Compute SVD
U, E, Vt = la.svd(PPMI, full_matrices=False)
E = np.diag(E)

# Verify reconstruction
print(np.allclose(PPMI, U @ E @ Vt))


V = Vt.T
reduced_PPMI = PPMI @ V[:, :3]
print()
print("Reduced PPMI is \n", reduced_PPMI)

#words = list(vocab)

# Compute distances on full PPMI matrix #REPLACE THIS WITH JUST PPMI
dist_women_men_full = norm(PPMI[f_list.index('women')] - PPMI[f_list.index('men')])
dist_women_dogs_full = norm(PPMI[f_list.index('women')] - PPMI[f_list.index('dogs')])
dist_men_dogs_full = norm(PPMI[f_list.index('men')] - PPMI[f_list.index('dogs')])
dist_feed_like_full = norm(PPMI[f_list.index('feed')] - PPMI[f_list.index('like')])
dist_feed_bite_full = norm(PPMI[f_list.index('feed')] - PPMI[f_list.index('bite')])
dist_like_bite_full = norm(PPMI[f_list.index('like')] - PPMI[f_list.index('bite')])

# Compute distances on reduced PPMI matrix
dist_women_men_reduced = norm(reduced_PPMI[f_list.index('women')] - reduced_PPMI[f_list.index('men')])
dist_women_dogs_reduced = norm(reduced_PPMI[f_list.index('women')] - reduced_PPMI[f_list.index('dogs')])
dist_men_dogs_reduced = norm(reduced_PPMI[f_list.index('men')] - reduced_PPMI[f_list.index('dogs')])
dist_feed_like_reduced = norm(reduced_PPMI[f_list.index('feed')] - reduced_PPMI[f_list.index('like')])
dist_feed_bite_reduced = norm(reduced_PPMI[f_list.index('feed')] - reduced_PPMI[f_list.index('bite')])
dist_like_bite_reduced = norm(reduced_PPMI[f_list.index('like')] - reduced_PPMI[f_list.index('bite')])

# Print distances
print('Distances on full matrix:')
print(dist_women_men_full, dist_women_dogs_full, dist_men_dogs_full, dist_feed_like_full, dist_feed_bite_full, dist_like_bite_full)

print('Distances on reduced matrix:')
print(dist_women_men_reduced, dist_women_dogs_reduced, dist_men_dogs_reduced, dist_feed_like_reduced, dist_feed_bite_reduced, dist_like_bite_reduced)


#PART 2!!!
print("\n\n\n PART 2 \n\n\n")


def load_word_vectors(file_path):
    word_vectors = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            word_vectors[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return word_vectors

def load_word_vectors_from_tar_gz(file_path):
    # extract all contents to a temporary folder
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path='./temp_folder')

    # find the file with word vectors in the temporary folder
    for root, dirs, files in os.walk('./temp_folder'):
        for file in files:
            if file.endswith('.txt'):
                word_vectors = load_word_vectors(os.path.join(root, file))
                break

    # os.system('rm -rf ./temp_folder')

    return word_vectors

  # load the synonym dataset
def load_synonyms(file_path):
    synonyms = []
    with open(file_path, 'r') as f:
        for line in f:
            word, synonym = line.strip().split('\t')
            if synonym != '0':  # ignore entries with no synonym
                synonyms.append((word[3:], synonym[3:]))
    return synonyms

# create the multiple-choice question set
def create_question_set(synonyms, word_vectors, num_questions=1000, num_choices=5):
    question_set = []
    for i in range(num_questions):
        word, correct_answer = random.choice(synonyms)
        distractors = []
        while len(distractors) < num_choices-1:
            word_pair = random.choice(synonyms)
            if word_pair[0] != word and word_pair[0] not in distractors:
                distractors.append(word_pair[0])
        choices = distractors + [correct_answer]
        random.shuffle(choices)
        question_set.append((word, choices))
    return question_set


# evaluate the multiple-choice question set using cosine similarity
def evaluate_cosine_similarity(question_set, word_vectors):
    num_correct = 0
    for word, choices in question_set:
        max_sim = -1
        best_choice = ''
        for choice in choices:
            if choice in word_vectors.keys() and word in word_vectors.keys():
                sim = 1 - cosine(word_vectors[word], word_vectors[choice])
                if sim > max_sim:
                    max_sim = sim
                    best_choice = choice
        if best_choice == choices[-1]:
            num_correct += 1
        #print(len(question_set))
    return num_correct / len(question_set)

# evaluate the multiple-choice question set using Euclidean distance
def evaluate_euclidean_distance(question_set, word_vectors):
    num_correct = 0
    for word, choices in question_set:
        min_dist = np.inf
        best_choice = ''
        for choice in choices:
            if choice in word_vectors.keys() and word in word_vectors.keys():
                dist = euclidean(word_vectors[word], word_vectors[choice])
                if dist < min_dist:
                    min_dist = dist
                    best_choice = choice
        if best_choice == choices[-1]:
            num_correct += 1
    return num_correct / len(question_set)

if __name__ == '__main__':
    # Load the synonym dataset
    synonym_file = "EN_syn_verb.txt"
    synonyms = load_synonyms(synonym_file)
    #print(synonyms)
    
    # Load the word vectors
    composes_file = "EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt.tar.gz"
    word2vec_file = "GoogleNews-vectors-rcv_vocab.txt.tar.gz"
    composes_vectors = load_word_vectors_from_tar_gz(composes_file)
    word2vec_vectors = load_word_vectors_from_tar_gz(word2vec_file)
    #test_comp = list(composes_vectors.keys())
    #test_w2 = list(word2vec_vectors.values())
    #print(type(composes_vectors))
    
    print("Number of word vectors loaded:", len(composes_vectors))
    print("Number of synonym pairs loaded:", len(synonyms))

    # Create the synonym multiple-choice question test set
    test_set_word2vec = create_question_set(synonyms, word2vec_vectors)
    #print(test_set_word2vec) #prints the synonym test set
    test_set_comp = create_question_set(synonyms, composes_vectors)
    #print(test_set_comp) #prints the synonym test set
    #print(test_set) #correct
    # Compute accuracy using Euclidean distance and cosine similarity
    composes_euclidean_acc = evaluate_euclidean_distance(test_set_comp, composes_vectors)
    composes_cosine_acc = evaluate_cosine_similarity(test_set_comp, composes_vectors)
    word2vec_euclidean_acc = evaluate_euclidean_distance(test_set_word2vec, word2vec_vectors)
    word2vec_cosine_acc = evaluate_cosine_similarity(test_set_word2vec, word2vec_vectors)

    # Print results
    print("Accuracy for COMPOSES with Euclidean distance: {:.2f}%".format(composes_euclidean_acc * 100))
    print("Accuracy for COMPOSES with cosine similarity: {:.2f}%".format(composes_cosine_acc * 100))
    print("Accuracy for word2vec with Euclidean distance: {:.2f}%".format(word2vec_euclidean_acc * 100))
    print("Accuracy for word2vec with cosine similarity: {:.2f}%".format(word2vec_cosine_acc * 100))
    df = pd.DataFrame({
    'Method': ['COMPOSES (Euclidean)', 'COMPOSES (cosine)', 'Google Word2Vec (Euclidean)', 'Google Word2Vec (cosine)'],
    'Accuracy': [composes_euclidean_acc*100, composes_cosine_acc*100, word2vec_euclidean_acc*100, word2vec_cosine_acc*100]
    })
    print(df)
        

#ANALOGY CODE!!!


composes_file = "EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt.tar.gz"
composes_vectors2 = load_word_vectors_from_tar_gz(composes_file)
# word2vec_file = "GoogleNews-vectors-rcv_vocab.txt.tar.gz"
# word2vec_vectors2 = load_word_vectors_from_tar_gz(word2vec_file)

def compute_vector_diff(word1, word2):
    if word1 not in composes_vectors2.keys() or word2 not in composes_vectors2.keys():
        return None
    return composes_vectors2[word1] - composes_vectors2[word2]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

questions_list = list()
q_bool = False
ql = list()
count = 0
with open("SAT-package-V3.txt", "r") as f:
    for lines in f:
        lines = lines.split()
        if len(lines) >= 2:
            lines = lines[:2]
       # print(lines)
        if len(lines) == 0:
            q_bool = True
        if q_bool == True and count <=7 and len(lines) != 0:
            if count == 0:
                count+=1
                continue
            q_temp = " ".join(lines)
           # print(q_temp)
            ql.append(q_temp)
            count+=1
            if count == 8:
                #print(ql)
                count =0
                questions_list.append(ql)
                ql = list()
                q_bool = False
    #print(questions_list)


# initialize variables
num_correct = 0
num_total = 0

# loop over questions

answer = ""
for ques in questions_list:
    #print(ques)
    max_sim = -1
    given1 = ques[0].split()[0]
    #print(given1)
    given2 = ques[0].split()[1]
    choices = ques[1:6]
    ans = ques[6]
    if ans == "a":
        ans =0
    if ans == "b":
        ans =1
    if ans == "c":
        ans =2
    if ans == "d":
        ans =3
    if ans == "e":
        ans =4
    #print(ans)
    #print(choices)
    given_vector = compute_vector_diff(given1, given2)
    if given_vector is None:
        continue

    for choi in choices:
        #print(ch.split())
        ch = choi.split()
        choice_vec = compute_vector_diff(ch[0], ch[1])
        if choice_vec is None:
            continue
        #print(given_vector)
        sim = cosine_similarity(given_vector, choice_vec)
        if sim > max_sim:
            max_sim = sim
            answer = choices.index(choi)
            #print(answer)
    if answer == ans:
        num_correct +=1
    num_total +=1
    
#print(num_total)
accuracy = num_correct / num_total
print("Accuracy: {:.2f}%".format(accuracy * 100))



