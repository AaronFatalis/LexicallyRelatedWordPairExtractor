# Aaron Vella - Automatic Lexical Relation Extractor

# Code able to run without providing a corpus and uses previously collected data
try:
    import MlrsReader.MlrsReader as MlrsReader
except:
    pass

from nltk.chunk.regexp import *
from collections import defaultdict
from operator import itemgetter
import math
import os
import json
import getpass

try:
    corpus_path = '../../CorpusSplit' # Change dir to data location
    reader = MlrsReader.MlrsReader(corpus_path, r'.*\.txt', ["words", "pos", "lemma", "root"])
except:
    no_data = True


def lookup(word1, word2):
    
    """
    Takes two words as parameters (strings) and iterates through the corpora
    to find them in the same sentence.
    """

    maxout = int(input("Ikteb numru ta' rizultati (-1 ghal-kollox): "))
    sents_list = []

    count = 0
    for sentence in reader.sents():
        if (word1 in sentence) and (word2 in sentence):
            sents_list.append(form_sent_str(sentence))
            count = count + 1
            number = count + 1
            if number == maxout:
                print()
                print("Aw taħt tista tara is-sentenzi li instabu: ")
                return sents_list
            
    print("Instabu " + str(count) + " eżempji fil-Korpus")
    return sents_list
    

####################################################


def form_sent_str(sentence):

    """
    Takes a sentence as a list and returns it into a string.
    """

    return " ".join(sentence)

####################################################


def seed_file_opener(directory):

    """
    Given the directory, opens the seed files as a list of tuples.
    """

    file = open(directory, 'r', encoding='utf-8')
    seeds = [(line.split('\t')) for line in file]
    return [(seed[0], seed[1].strip('\n')) for seed in seeds]

####################################################


def form_patterns():

    """
    Looks for all occurances of the seed words and saves patterns to file
    WARNING - TAKES MORE THAN 3 HOURS TO COMPLETE
    Will be modified soon to ask user if they want to replace the old files or not.
    """
    
    antonym_file = 'Patterns/Antonym_Patterns.txt'
    hyperhypo_file = 'Patterns/Hyper-Hypo_Patterns.txt'
    meronym_file = 'Patterns/Meronym_Patterns.txt'
    
    antonym_seeds = seed_file_opener('Seeds/Antonym_Seeds.txt')
    
    hyperhypo_seeds = seed_file_opener('Seeds/Hyper-Hypo_Seeds.txt')

    meronym_seeds = seed_file_opener('Seeds/Meronym_Seeds.txt')

    relation_lists = [(antonym_seeds, antonym_file),
                      (hyperhypo_seeds, hyperhypo_file),
                      (meronym_seeds, meronym_file)]

    for seeds, file_dir in relation_lists:
        with open(file_dir, 'w', encoding='utf-8') as outfile:
            print('Starting New Lexical Relation ...')
            count = 0
            count2 = 0
            count3 = 0
            for sentence in reader.tagged_lem_sents():
                count2 = count2 + 1
                a = count2 % 20000
                if a == 0:
                    count3 = count3 + 1
                    print('This will print for every 20000 sentences ' + str(count3))
                for word1, word2 in seeds:
                    tagged_sentence = [list(sent_tags) for sent_tags in zip(*sentence)]
                    try:
                        if (word1 in tagged_sentence[0]) and (word2 in tagged_sentence[0]):
                            outfile.write('\t'.join([word1, word2, form_sent_str(tagged_sentence[0]),
                                                     form_sent_str(tagged_sentence[1])]) + "\n")
                            count = count + 1
                            print(count)
                    except IndexError:
                        pass
            print("Instabu " + str(count))
                
    print("Success")
    return

####################################################


def load_patterns(directory):

    """
    Loads patterns in a list, with the following content:
    [word1,word2,tokenised sentence + postags, postags]
    
    [0][1] = word1 and word2
    [2] = whole sentence in list format
    [3] = list of pos tags which make up the sentence
    [4] = list of tuples containing both word and postag
    """
    
    file = open(directory, 'r', encoding='utf-8')
    items = [(line.split('\t')) for line in file]
    clean_list = [(item[0], item[1], item[2].split(), item[3].strip('\n')) for item in items]
    clean_list = [(item[0], item[1], item[2].split(), item[3].split()) for item in items]
    
    finlist = []
    for i in clean_list:
        finlist.append([i[0], i[1], i[2], i[3], [(word, pos) for word, pos in zip(i[2], i[3])]])
    
    return finlist

####################################################


def pattern_counting(directory):
    """
    To be used with the output of generate_sent_regex(directory)
    Loads in file, computes a Frequency Distribution of the patterns,
    sorts by most common and saves to a new file.
    """

    open_patterns = open(directory)
    patterns = [pattern.strip('\n') for pattern in open_patterns]
    save_to_dir = 'FreqDist Patterns/' + directory.strip('Pattern Parsing/')

    pattern_frequency = dict()

    for pattern in patterns:
        if pattern not in pattern_frequency:
            pattern_frequency[pattern] = 1
        else:
            pattern_frequency[pattern] += 1

    filt_pattern_frequency = {x: pattern_frequency[x] for x in pattern_frequency if pattern_frequency[x] not in {1,2,3}}

    ss = ''
    for pattern in sorted(filt_pattern_frequency, key=filt_pattern_frequency.get, reverse=True):
        ss = ss + pattern + '\t' + str(filt_pattern_frequency[pattern]) + '\n'

    ss = ss + 'Total number of patterns:' + '\t' + str(sum(filt_pattern_frequency.values()))

    with open(save_to_dir, 'w', encoding='utf-8') as file:
        file.write(ss)

    return
        

####################################################

def generate_sent_regex(directory, relation):
    """
    Filters patterns acquired from form_pattern() to only include patterns with
    same pos tag target words and saves to file the patterns to be used with the
    nltk chunker.
    """
    filename = directory.strip('Patterns/')

    # simple rule to locate target words and parser depending on relation + consider only relevant pos tags
    if relation == 'A':  # Antonymy (word after second seed is mostly irrelevant)
        simple_rule = ChunkRule('<.+><X><.*>+<X>', 'PATTERN')
        pos = 'ADJ'
    elif relation == 'H':  # Hyper / Hypo (words such as 'iehor' at the end)
        simple_rule = ChunkRule('<X><.*>+<X><.+>', 'PATTERN')
        pos = 'NOUN'
    elif relation == 'M':  # Meronymy (nothing seems relevant outside of window)
        simple_rule = ChunkRule('<X><.*>+<X>', 'PATTERN')
        pos = 'NOUN'

    parser = RegexpChunkParser([simple_rule], simple_rule.descr())
    
    data = load_patterns(directory)
    print('Filtering and adding content to data...')
    tag_patterns = []

    for tagsent in data:

        word1 = tagsent[0]
        word2 = tagsent[1]

        word1_index = tagsent[2].index(word1)
        word2_index = tagsent[2].index(word2)

        # checks if both target words have the same postag - else ignore
        if (tagsent[4][word1_index][1] == tagsent[4][word2_index][1]) and (tagsent[4][word1_index][1] == pos):
            
            tagsent[4][word1_index] = list(tagsent[4][word1_index])
            tagsent[4][word2_index] = list(tagsent[4][word2_index])

            tagsent[4][word1_index][1] = 'X'  # ~
            tagsent[4][word2_index][1] = 'X'  # ~

            tagsent[4][word1_index] = tuple(tagsent[4][word1_index])
            tagsent[4][word2_index] = tuple(tagsent[4][word2_index])
        
            parse_tree = parser.parse(tagsent[4])

            for tree in parse_tree.subtrees():
                tree = tree.leaves()
                tree_tags = [tag for word, tag in tree]
                tag_patterns.append(''.join(['<' + tag + '>' for tag in tree_tags]))
    
    print('Saving to File...')
    ss = ''
    for pattern in tag_patterns:
        ss = ss + str(pattern) + '\n'
    
    with open('Pattern Parsing/' + filename + '.txt', 'w', encoding='utf-8') as f:
        f.write(ss)
                    
####################################################


def run_pattern_conversion():
    """
    Runs generate_sent_regex() and pattern_counting() sequentially for all
    lexical relations
    """
    
    rels = [('Patterns/Antonym_Patterns.txt', 'A'),
            ('Patterns/Hyper-Hypo_Patterns.txt', 'H'),
            ('Patterns/Meronym_Patterns.txt', 'M')]
    
    for rel in rels:
        generate_sent_regex(rel[0], rel[1])
    
    dirs_to_count = ['Pattern Parsing/Antonym_Patterns.tx.txt',
                     'Pattern Parsing/Hyper-Hypo_Patterns.tx.txt',
                     'Pattern Parsing/Meronym_Patterns.tx.txt']

    for d in dirs_to_count:
        pattern_counting(d)
        
    return

####################################################


def is_sublist_of(pattern, sentence):
    """
    Checks if a list is a sublist of another (returns boolean)
    """
    
    # takes the index of each pos tag which is the same as the start of the pattern
    pt_start = [k for k, j in enumerate(sentence) if j == pattern[0]]
    pt_len = len(pattern)  # pattern length

    for possible_start in pt_start:
        curr_check = sentence[possible_start: possible_start + pt_len]

        if pattern == curr_check:
            return True

    return False

####################################################


def extract_sublists(pattern, sentence, words):
    """
    Function which extracts all sublists which are the same to the given list
    in another list
    """
    pt_start = [k for k, j in enumerate(sentence) if j == pattern[0]]
    pt_len = len(pattern)  # pattern length
    pattern_matches_list = []

    for possible_start in pt_start:
        curr_check = sentence[possible_start: possible_start + pt_len]

        if pattern == curr_check:
            pattern_matches_list.append(words[possible_start: possible_start + pt_len])

    return pattern_matches_list

####################################################


def ext_sent_analysis():
    """
    Acquires pmi values for seed words
    """
    relations = ['Antonym', 'Hyper-Hypo', 'Meronym']

    for relation in relations:

        open_patterns = open('FreqDist Patterns/' + relation + '_Patterns.tx.tx')
        patterns = [pattern.split()[0] for pattern in open_patterns]
        patterns = patterns[:-1]

        if relation == 'Antonym':
            cond = ('<X>', '<ADJ>')
        else:
            cond = ('<X>', '<NOUN>')

        seeds = seed_file_opener('Seeds/'+relation+'_Seeds.txt')

        file = load_patterns('Patterns/' + relation + '_Patterns.txt')

        for seed1, seed2 in seeds:

            print('Processing seeds: ' + seed1 + ' ' + seed2)

            file_subset = [subset for subset in file if (subset[0] == seed1) and (subset[1] == seed2)]

            pmi_b = len(file_subset)
            print(str(pmi_b) + ' sentences containing these seeds in Corpus')
            ss = ''

            for pattern in patterns:

                # convert pattern's content into parsable patterns by replacing postags
                conv_pattern = pattern.replace(cond[0], cond[1])
                conv_pattern = conv_pattern.split('<')[1:]  # ['DEF>', 'NOUN>', 'DEF>', 'NOUN>']
                conv_pattern = [a.replace('>', '') for a in conv_pattern]  # ['DEF', 'NOUN', 'DEF', 'NOUN']

                pmi_a = 0

                for sent in file_subset:
                    curr_sent = sent[3]

                    if is_sublist_of(conv_pattern, curr_sent):
                        pmi_a += 1

                curr_line = pattern + '\t' + str(pmi_a) + '\t' + str(pmi_b) + '\n'
                ss = ss + curr_line

            with open('SeedPatternFreq/'+relation+'_pmiAB_' + seed1 + '-' + seed2 + '.txt', 'w', encoding='utf-8') as f:
                f.write(ss)

    return 'Success'

####################################################


def patterns_in_corpus(relation):
    """
    Extracts pattern success frequency and new word pairs which fulfill the patterns
    """
    successful_parse = dict()
    new_word_patterns = defaultdict(set)
    new_word_freqs = dict()
    open_patterns = open('FreqDist Patterns/' + relation + '_Patterns.tx.tx')
    patterns = [pattern.split()[0] for pattern in open_patterns]
    patterns = patterns[:-1]

    if relation == 'Antonym':
        cond = ('<X>', '<ADJ>')
    else:
        cond = ('<X>', '<NOUN>')

    sent_count = 0
    
    tot_sents = 10257226
    one_percent = round(tot_sents / 100)

    for sent_pos in reader.tagged_lem_sents():
        sent_count += 1
        if sent_count % one_percent == 0:
            print(str(round((sent_count / tot_sents) * 100)) + '% Complete...')
        for pattern in patterns:
            # convert pattern to list to find index of 'X'
            pat_to_list = pattern.split('>')[:-1]
            target_word_index = [k for k, j in enumerate(pat_to_list) if j == '<X']

            # convert pattern's content into parsable patterns by replacing postags
            conv_pattern = pattern.replace(cond[0], cond[1])
            conv_pattern = conv_pattern.split('<')[1:]  # ['DEF>', 'NOUN>', 'DEF>', 'NOUN>']
            conv_pattern = [a.replace('>', '') for a in conv_pattern]  # ['DEF', 'NOUN', 'DEF', 'NOUN']

            words = [word for word, tags in sent_pos]
            pos_tags = [tags for word, tags in sent_pos]

            if is_sublist_of(conv_pattern, pos_tags):

                for sent_pattern in extract_sublists(conv_pattern, pos_tags, words):

                    # keep record of pattern's success
                    if pattern in successful_parse:
                        successful_parse[pattern] += 1
                    else:
                        successful_parse[pattern] = 1

                    # store new words with found pattern
                    new_word1 = sent_pattern[target_word_index[0]]
                    new_word2 = sent_pattern[target_word_index[1]]
                    new_word_patterns[(new_word1, new_word2)].add(pattern)
                    
                    # store new word pair frequencies
                    if (new_word1, new_word2) in new_word_freqs:
                        new_word_freqs[(new_word1, new_word2)] += 1
                    else:
                        new_word_freqs[(new_word1, new_word2)] = 1

    # save to file
    print('Saving to File...')
    
    with open('PatternSuccessFreq/' + relation + '_pmiC.txt', 'w', encoding='utf-8') as f:
        for item in sorted(successful_parse, key=successful_parse.get, reverse=True):
            f.write(item + '\t' + str(successful_parse[item]) + '\n')
    
    with open('NonSeedMatches/' + relation + '_Words.txt', 'w', encoding='utf-8') as f:
        for item in new_word_patterns.items():
            f.write(item[0][0] + '\t' + item[0][1] + '\t' + str(item[1]) + '\n')

    with open('NonSeedMatchesFreq/' + relation + '_Words.txt', 'w', encoding='utf-8') as f:
        for item in sorted(new_word_freqs, key=new_word_freqs.get, reverse=True):
            f.write(item[0] + '\t' + item[1] + '\t' + str(new_word_freqs[item]) + '\n')
    
    return 'Success'

####################################################


def patterns_in_corpus_init():
    """
    Launch patterns_in_corpus().
    """
    relations = ['Antonym', 'Hyper-Hypo', 'Meronym']
    for relation in relations:
        print('Starting ' + relation + '...')
        patterns_in_corpus(relation)
    return

####################################################

def get_new_word_pair_pmi():
    """
    Get values required to evaluate the pmi of newly found word pairs
    WARNING: only to be used after patterns_in_corpus() function.
    """
    for relation in ['Antonym', 'Hyper-Hypo', 'Meronym']:
        print(relation)
        
        word_pair_frequency = dict()
        pattern_frequency = dict()

        open_patterns = open('FreqDist Patterns/' + relation + '_Patterns.tx.tx')
        patterns = [pattern.split()[0] for pattern in open_patterns]
        patterns = patterns[:-1]

        if relation == 'Antonym':
            cond = ('<X>', '<ADJ>')
            cut_off = 10
        elif relation == 'Hyper-Hypo':
            cond = ('<X>', '<NOUN>')
            cut_off = 100
        else:
            cond = ('<X>', '<NOUN>')
            cut_off = 300
        
        tot_sents = 10257226
        sent_count = 0
        one_percent = round(tot_sents / 100)
        
        with open('NonSeedMatchesFreq/' + relation + '_Words_Clean.txt', 'r', encoding='utf-8') as f:
            new_word_pairs = [line.split() for line in f if int(line.split()[2]) >= cut_off]
            
        for corpus_sent in reader.tagged_lem_sents():
            sent_count += 1
            if sent_count % one_percent == 0:
                print(str(round((sent_count / (tot_sents)) * 100)) + '% Complete...')
            for words in new_word_pairs:
                sent = [word for word, pos in corpus_sent]
                if words[0] in sent and words[1] in sent:
                    if (words[0], words[1]) in word_pair_frequency:
                        word_pair_frequency[(words[0], words[1])] += 1
                    else:
                        word_pair_frequency[(words[0], words[1])] = 1
                        
                    sent_pos = [pos for word, pos in corpus_sent]
                    
                    for pattern in patterns:
                        conv_pattern = pattern.replace(cond[0], cond[1])
                        conv_pattern = conv_pattern.split('<')[1:]
                        conv_pattern = [a.replace('>', '') for a in conv_pattern]
                        
                        if is_sublist_of(conv_pattern, sent_pos):
                            for sent_pattern in extract_sublists(conv_pattern, sent_pos, sent):
                                joined_tuple = words[0] + " " + words[1]
                                if joined_tuple in pattern_frequency:
                                    if pattern in pattern_frequency[joined_tuple]:
                                        pattern_frequency[joined_tuple][pattern] += 1
                                    else:
                                        pattern_frequency[joined_tuple][pattern] = 1
                                else:
                                    pattern_frequency[joined_tuple] = {pattern : 1}
                 
        with open('NewWordPairPMI/' + relation + '.json', 'w') as jsf:
            json.dump(pattern_frequency, jsf)        
        with open('NonSeedMatchesTotFreq/' + relation + '_Words.txt', 'w', encoding='utf-8') as f:
            for item in sorted(word_pair_frequency, key=word_pair_frequency.get, reverse=True):
                f.write(item[0] + '\t' + item[1] + '\t' + str(word_pair_frequency[item]) + '\n')
        
            
    return


####################################################

def cleaning_new_words(directory):
    """
    Takes the new word pair frequency files and removes pairs which are
    the same while adding up pairs which are the same.
    """
    word_list = []
    clean_word_dict = dict()
    with open(directory, 'r', encoding='utf-8') as file:
        for line in file:
            line.strip('\n')
            word_list.append(line.split())

    for word1, word2, frequency in word_list:
        if word1 == word2:
            pass
        elif ((word1, word2) not in clean_word_dict) and ((word2, word1) in clean_word_dict):
            clean_word_dict[(word2, word1)] += int(frequency)
        else:
            clean_word_dict[(word1, word2)] = int(frequency)
    
    with open(directory[:-4] + '_Clean.txt', 'w', encoding='utf-8') as f:
        for item in sorted(clean_word_dict, key=clean_word_dict.get, reverse=True):
            f.write(item[0] + '\t' + item[1] + '\t' + str(clean_word_dict[item]) + '\n')
        
    return 'Cleaning Complete'


####################################################

def cleaning_new_words_all():
    """
    Runs cleaning_new_words() for all lexical relations.
    """
    for relation in ['Antonym', 'Hyper-Hypo', 'Meronym']:
        print('Starting ' + relation + '...')
        cleaning_new_words('NonSeedMatchesFreq/' + relation + '_Words.txt')
    return

####################################################


def calculate_pmi(pmi_a, pmi_b, pmi_c):
    """
    Takes three values to return the pmi.
    """
    try:
        return math.log2(pmi_a / (pmi_b * pmi_c))
    except:
        return 0

####################################################


def run_espresso():
    """
    load in data aquired from previous functions which acquire pmi values,
    and uses them to give a reliability measure to each pattern.
    """
    for relation in ['Antonym', 'Hyper-Hypo', 'Meronym']:
        print(relation)
        
        pattern_pmi_list = []
        seed_rel_dict = dict()
        fin_rel_dict = dict()
        
        # load in data
        file_names = [name for name in os.listdir('SeedPatternFreq') if relation in name]
        pattern_a_b_values = []
        for file in file_names:
            pat_list = []
            with open('SeedPatternFreq/' + file, 'r', encoding='utf-8') as f:
                for line in f:
                    pat_list.append(line)
                pattern_a_b_values.append((file.split('_')[-1:][0][:-4], pat_list))
        # pattern_a_b_values : [[patterns - antik / modern],[patterns - iswed / abjad], ...]

        pattern_list = [patterns.split()[0] for patterns in pattern_a_b_values[0][1]]
        
        c_values = dict()
        with open('PatternSuccessFreq/' + relation + '_pmiC.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split()
                c_values[line[0]] = line[1]
                
        max_pmi = 0
        
        for words, patterns in pattern_a_b_values:
            word_pattern_pmi_list = []
            for line in patterns:
                current_line = line.split()
                
                pattern = current_line[0]
               
                a_value = int(current_line[1])
                b_value = int(current_line[2])
                c_value = int(c_values[pattern])
                
                pmi_value = calculate_pmi(a_value, b_value, c_value) + 1
                word_pattern_pmi_list.append((words, pattern, pmi_value))
                largest_pmi = max(word_pattern_pmi_list)
                
                if largest_pmi[2] > max_pmi:
                    max_pmi = largest_pmi[2]
                
            pattern_pmi_list.append(word_pattern_pmi_list)
            
        num_of_patterns = len(patterns)
        
        # seed reliability
        for pattern_count, pattern_file in enumerate(pattern_pmi_list[0]):
            current_pattern_subset = []
            for file in range(0, len(pattern_pmi_list)):
                current_pattern_subset.append(pattern_pmi_list[file][pattern_count])

            seed_reliability = 0
            for seeds, pattern, pmi in current_pattern_subset:
                seed_reliability += pmi/max_pmi

            seed_rel_dict[pattern] = seed_reliability / 30

        with open('NewWordPairPMI/' + relation + '.json') as jsonfile:
            new_word_values = json.load(jsonfile)

        new_word_b_values = dict()
        with open('NonSeedMatchesTotFreq/' + relation + '_Words.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split()
                new_word_b_values[line[0] + ' ' + line[1]] = line[2]
            
        # new_word_pairs_pmi
        pattern_pmi_list = []
        
        for word_pairs in new_word_values:
            word_pattern_pmi_list = []
            #for pattern in new_word_values[word_pairs]:
            for pattern in pattern_list:
                try:
                    pmi_value = calculate_pmi(new_word_values[word_pairs][pattern],
                                          int(new_word_b_values[word_pairs]),
                                          int(c_values[pattern])) + 1
                except KeyError:
                    
                    pmi_value = calculate_pmi(0,
                                          int(new_word_b_values[word_pairs]),
                                          int(c_values[pattern])) + 1
                
                word_pattern_pmi_list.append((word_pairs, pattern, pmi_value))
                largest_pmi = max(word_pattern_pmi_list)
                
                if largest_pmi[2] > max_pmi:
                    max_pmi = largest_pmi[2]
                
            pattern_pmi_list.append(word_pattern_pmi_list)
            
        # new_word_pairs_reliability
        
        for pattern_count, pattern_file in enumerate(pattern_pmi_list[0]):
            current_pattern_subset = []
            for file in range(0, len(pattern_pmi_list)):
                current_pattern_subset.append(pattern_pmi_list[file][pattern_count])
            seed_reliability = 0
            for words, pattern, pmi in current_pattern_subset:
                seed_reliability += ((pmi/max_pmi) * seed_rel_dict[pattern])

            fin_rel_dict[pattern] = seed_reliability/num_of_patterns

        esp_rel_words_pats = dict()
        
        for words in new_word_values.keys():
            for pattern in new_word_values[words].keys():
                if words in esp_rel_words_pats:
                    esp_rel_words_pats[words][pattern] = fin_rel_dict[pattern]
                else:
                    esp_rel_words_pats[words] = {pattern : fin_rel_dict[pattern]}              
                
        # save each word pair with reliability scale        
        
        with open('Espresso_Patterns/' + relation + '_rm.txt', 'w', encoding = 'utf-8') as f: #save pattern with reliability rating
            for pat in sorted(fin_rel_dict, key=fin_rel_dict.get, reverse=True):
                f.write(pat + ' ' + str(fin_rel_dict[pat]) + '\n')

        with open('Espresso_Patterns/' + relation + '_rm.json', 'w') as jsf:
            json.dump(esp_rel_words_pats, jsf)
    
    return 'Complete!'
            

####################################################

def return_relations(word):
    """
    Given a word, the function returns other words which could relate to
    it in a given relation
    """
    
    relations = ['Antonym', 'Hyper-Hypo', 'Meronym']
    
    print('Ma liema relazzjonijiet tixtieq tikkumpara din il-kelma?')
    for num, relation in enumerate(relations):
        print(str(num) + '. ' + relation)
    print('Daħħal li trid jew afas enter jekk trid kollox')
    try:
        ch = int(input("Ikteb l' għażla tiegħek: "))
    except ValueError:
        ch = 99
    print()

    if ch == 0:
        relations = ['Antonym']
    elif ch == 1:
        relations = ['Hyper-Hypo']
    elif ch == 2:
        relations = ['Meronym']

    for relation in relations:
        print("Issortjajt bil-Frekwenza tal-kliem: ")
        
        with open('NonSeedMatchesFreq/' + relation + '_Words_Clean.txt', 'r', encoding='utf-8') as f:
            words_in_relation = []
            for line in f:
                line = line.split()
                if len(words_in_relation) == 5:
                    break
                elif word == line[0]:
                    words_in_relation.append(line[1])
                elif word == line[1]:
                    words_in_relation.append(line[0])
            print((relation, words_in_relation))
        print("Issortjajt b'affidabilità ta\' l-Espresso: ")

        with open('Espresso_Patterns/' + relation + '_rm.json') as jsonfile:
            esp_words = json.load(jsonfile)

        sub_dict = {words : esp_words[words] for words in esp_words if word in words.split()}
        list_of_rels = []
        for words in sub_dict:
            total_reliability = 0
            for pattern in sub_dict[words]:
                total_reliability += sub_dict[words][pattern]
            list_of_rels.append((words.split(), total_reliability))
            
        out_list = []
        for i in sorted(list_of_rels, key=itemgetter(1), reverse =True)[0:6]:
            if word == i[0][0]:
                out_list.append(i[0][1])
            else:
                out_list.append(i[0][0])
            
        print(out_list)

    return

####################################################

def sub_menu():
    """
    Sub Menu
    """
    print('1. Iġġenera Mudelli Ġodda u Naddaf ir-Riżultati')
    print('2. Uża il-Mudelli Biex Issib Kliem Ġodda u Aħdem Statistiċi tal-PMI u Espresso')
    print('3. Oħroġ fil-Menu ta\' Barra')
    print()
    choice = int(input('Jekk jogħġbok għażel waħda mil-funzjonijiet: '))
    if choice == 1:
        form_patterns()
        run_pattern_conversion()
    elif choice == 2:
        ext_sent_analysis()
        patterns_in_corpus_init()
        cleaning_new_words_all()
        get_new_word_pair_pmi()
        run_espresso()
    elif choice == 3:
        menu()
        return

####################################################
    
def menu():
    """
    Main Menu
    """

    print('Estrazzjoni ta\' Relazzjonijiet Lessikali Bil-Malti v 1.0.2 - Aaron Vella')
    print('_________________________________________________________________________')
    print('')
    print('1. Fittex Kelma')
    print('2. Iġġenera Data Ġdida')
    print('3. Instruzzjonijiet')
    print('4. Għalaq Il-Program')
    print()
    choice = int(input('Jekk jogħġbok għażel waħda mil-funzjonijiet: '))

    while True:
        if choice == 1:
            word_input = input('Ikteb il-kelma li tixtieq tfittex: ')
            return_relations(word_input)
            print()
            choice = int(input('Jekk jogħġbok għażel waħda mil-funzjonijiet: '))
        elif choice == 2:
            if no_data:
                print('Korpus Mux Prezenti - jekk jogħġbok kellem administratur')
            else:
                print('Ikteb il-password biex tkompli: ')
                pass_req = getpass.getpass()
                if pass_req == 'admin':
                    sub_menu()
                else:
                    print('Il-password li ktibt mhux tajba')
            choice = int(input('Jekk jogħġbok għażel waħda mil-funzjonijiet: '))
        elif choice == 3:
            with open('Instuzzjonijiet.txt', 'r', encoding='utf-8') as f:
                      for line in f:
                          print(line.strip('\n'))
            print()
            choice = int(input('Jekk jogħġbok għażel waħda mil-funzjonijiet: '))
        elif choice == 4:
            print('Grazzi talli użajt din is-sistema.')
            print('Jekk għandek xi suġġerimenti li jistu isiru, ibgħat e-mail lil:')
            print('aaron.vella.995@gmail.com')
            input('Għafas ENTER biex tagħlaq il-programm')
            exit()
        else:
            choice = int(input('In-numru li daħħalt mhuwiex validu, jekk jogħġbok erġa prova: '))


if __name__ == '__main__':
    menu()
    
