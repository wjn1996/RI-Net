# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:30:25 2019

@author: Majigsuren Enkhsaikhan
"""
import os
import pandas as pd
import re
import spacy
from spacy.attrs import intify_attrs
nlp = spacy.load("en_core_web_sm")

import neuralcoref

import networkx as nx
# import matplotlib.pyplot as plt

#nltk.download('stopwords')
from nltk.corpus import stopwords
all_stop_words = ['many', 'us', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                  'today', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                  'september', 'october', 'november', 'december', 'today', 'old', 'new']
all_stop_words = sorted(list(set(all_stop_words + list(stopwords.words('english')))))

abspath = os.path.abspath('') ## String which contains absolute path to the script file
#print(abspath)
os.chdir(abspath)

### ==================================================================================================
# Tagger

def get_tags_spacy(nlp, text):
    doc = nlp(text) # 生成词对象
    entities_spacy = [] # Entities that Spacy NER found
    for ent in doc.ents: # doc.ents表示每个token的实体识别结果
        entities_spacy.append([ent.text, ent.start_char, ent.end_char, ent.label_])# entities_spacy：句子中所有实体列表，每个token对应的文本、起始以及实体标记。例子： [['June 2005', 48, 57, 'DATE'], ['second', 76, 82, 'ORDINAL'], ["B'Day", 95, 100, 'ORG'], ['2006', 102, 106, 'DATE'], ['Beyoncé', 895, 902, 'PERSON'], ['2013', 904, 908, 'DATE']]
    return entities_spacy

def tag_all(nlp, text, entities_spacy):
    if ('neuralcoref' in nlp.pipe_names):
        nlp.pipeline.remove('neuralcoref')    
    neuralcoref.add_to_pipe(nlp) # Add neural coref to SpaCy's pipe    将指代消解加入spacy的pipeline中
    doc = nlp(text)
    return doc

def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps因为是将实体和词性为名词的词直接汇总，可能有的实体也是名词，而出现重复或覆盖问题，则需要过滤掉这些重复的部分；例如输入的span可能有81个实体和名词，处理后输出则减少为50个
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result

def tag_chunks(doc):
    spans = list(doc.ents) + list(doc.noun_chunks) # 将实体识别得到的实体和所有词性为名词的token进行汇总
    spans = filter_spans(spans) # 因为实体和名词直接混合，会有重复，因此在此进行过滤
    with doc.retokenize() as retokenizer: # doc.retokenize()可以重新获取这些词在原始的句子中的位置等迭代信息
        string_store = doc.vocab.strings
        for span in spans: # 每个span代表一个token或实体，可以将其重新当做一个spacy对象处理，并获得相应属性
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': 'ENTITY'}, string_store))

def tag_chunks_spans(doc, spans, ent_type):
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': ent_type}, string_store))

def clean(text):
    # 文本清理
    text = text.strip('[(),- :\'\"\n]\s*')
    text = text.replace('—', ' - ')
    text = re.sub('([A-Za-z0-9\)]{2,}\.)([A-Z]+[a-z]*)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.)(\"\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.\/)(\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z]{3,}\.)([A-Z]+[a-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.)([A-Za-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    
    text = re.sub('’', "'", text, flags=re.UNICODE)           # curly apostrophe
    text = re.sub('‘', "'", text, flags=re.UNICODE)           # curly apostrophe
    text = re.sub('“', ' "', text, flags=re.UNICODE)
    text = re.sub('”', ' "', text, flags=re.UNICODE)
    text = re.sub("\|", ", ", text, flags=re.UNICODE)
    text = text.replace('\t', ' ')
    text = re.sub('…', '..', text, flags=re.UNICODE)           # elipsis
    text = re.sub('â€¦', '..', text, flags=re.UNICODE)
    text = re.sub('â€“', '-', text)           # long hyphen
    text = re.sub('\s+', ' ', text, flags=re.UNICODE).strip()
    text = re.sub(' – ', ' . ', text, flags=re.UNICODE).strip()

    return text

def tagger(text):  
    df_out = pd.DataFrame(columns=['Document#', 'Sentence#', 'Word#', 'Word', 'EntityType', 'EntityIOB', 'Lemma', 'POS', 'POSTag', 'Start', 'End', 'Dependency'])
    corefs = [] # 保存所有指代的词
    # text = clean(text) # 文本清理
    
    nlp = spacy.load("en_core_web_sm")
    entities_spacy = get_tags_spacy(nlp, text) # 获得每个token的实体识别结果
    #print("SPACY entities:\n", ([ent for ent in entities_spacy]), '\n\n')
    document = tag_all(nlp, text, entities_spacy) # 融入共指消解工具
    tokens = []
    for token in document:
        # if token.text.strip() != '':
        tokens.append([token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_])

    ### Coreferences
    # 
    if document._.has_coref: # 如果存在共指消解的词，则进行处理
        for cluster in document._.coref_clusters:
            main = cluster.main # 共指的词
            for m in cluster.mentions: # 所有指代的词（即原始的词，例如he，its，包括其本身）
                if (str(m).strip() == str(main).strip()): # 如果是其本身，则跳过
                    continue
                corefs.append([str(m), str(main)]) # 将所有指代的词加入corefs列表，m表示文本中原始的词，main表示原始对应的指代的词
    tag_chunks(document)
    
    # chunk - somethin OF something 名词分块，将相邻的两个名词或实体作为一个整体
    spans_change = []
    for i in range(2, len(document)):
        w_left = document[i-2]
        w_middle = document[i-1]
        w_right = document[i]
        if w_left.dep_ == 'attr':
            continue
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY' and (w_middle.text == 'of'): # or w_middle.text == 'for'): #  or w_middle.text == 'with'
            spans_change.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change, 'ENTITY') # 新合并生成的实体再次与现有的进行融合和过滤
    
    # chunk verbs with multiple words: 'were exhibited' 动词分块，相邻的两动词则可以合并为一个
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: verb + adp; verb + part
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'ADP' or w_right.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: adp + verb; part  + verb
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_right.pos_ == 'VERB' and (w_left.pos_ == 'ADP' or w_left.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')
    
    # chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk all between LRB- -RRB- (something between brackets)
    start = 0
    end = 0
    spans_between_brackets = []
    for i in range(0, len(document)):
        if ('-LRB-' == document[i].tag_ or r"(" in document[i].text):
            start = document[i].i
            continue
        if ('-RRB-' == document[i].tag_ or r')' in document[i].text):
            end = document[i].i + 1
        if (end > start and not start == 0):
            span = document[start:end]
            try:
                assert (u"(" in span.text and u")" in span.text)
            except:
                pass
                #print(span)
            spans_between_brackets.append(span)
            start = 0
            end = 0
    tag_chunks_spans(document, spans_between_brackets, 'ENTITY')
            
    # chunk entities  两个实体相邻时，合并
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'ENTITY')
    
    doc_id = 1
    count_sentences = 0
    prev_dep = 'nsubj'
    for token in document:
        # if token.text.strip() != '':
        if (token.dep_ == 'ROOT'):
            if token.pos_ == 'VERB':
                #  将pipeline的输出保存到csv，列名：['Document#', 'Sentence#', 'Word#', 'Word', 'EntityType', 'EntityIOB', 'Lemma', 'POS', 'POSTag', 'Start', 'End', 'Dependency']
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]
            else:
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, prev_dep]
        else:
            df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]

        if (token.text == '.'):
            count_sentences += 1
        prev_dep = token.dep_
    # df_out：以实体层面上的信息，corefs实体层面上的消解，tokens：token级别的信息
    return df_out, corefs, tokens

### ==================================================================================================
### triple extractor

def get_predicate(s):
    pred_ids = {}
    for w, index, spo in s:
        if spo == 'predicate' and w != "'s" and w != "\"": #= 11.95
            pred_ids[index] = w
    predicates = {}
    for key, value in pred_ids.items():
        predicates[key] = value
    return predicates # predicate谓词

def get_subjects(s, start, end, adps):
    subjects = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'subject' in spo or 'entity' in spo or 'object' in spo:
                subjects[index] = w
    return subjects
    
def get_objects(s, start, end, adps):
    objects = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'object' in spo or 'entity' in spo or 'subject' in spo:
                objects[index] = w
    return objects

def get_positions(s, start, end):
    adps = {}
    for w, index, spo in s:        
        if index >= start and index <= end:
            if 'of' == spo or 'at' == spo:
                adps[index] = w
    return adps

def create_triples(df_text, corefs):
    ## 创建三元组
    sentences = [] # 所有句子
    aSentence = [] # 某个句子
    
    for index, row in df_text.iterrows(): # 遍历每个token，index为索引，row为token的对象
        d_id, s_id, word_id, word, ent, ent_iob, lemma, cg_pos, pos, start, end, dep = row.items()
        if 'subj' in dep[1]:
            aSentence.append([word[1], word_id[1], 'subject']) # word等是一个元组，第一个元素为对应的特征“word”，第二个元素为对应的值，例如（“word”，“xxx”）
        elif 'ROOT' in dep[1] or 'VERB' in cg_pos[1] or pos[1] == 'IN':
            aSentence.append([word[1], word_id[1], 'predicate'])
        elif 'obj' in dep[1]:
            aSentence.append([word[1], word_id[1], 'object'])
        elif ent[1] == 'ENTITY':
            aSentence.append([word[1], word_id[1], 'entity'])        
        elif word[1] == '.':
            sentences.append(aSentence)
            aSentence = []
        else:
            aSentence.append([word[1], word_id[1], pos[1]])
    # sentences：按照实体情况保存整个句子token、index以及词性。例如： [[['Following', 0, 'predicate'], ["the disbandment of Destiny's Child", 1, 'object'], ['in', 2, 'predicate'], ['June 2005', 3, 'object'], [',', 4, ','], ['she', 5, 'subject'], ['released', 6, 'predicate'], ['her second solo album', 7, 'object'], [',', 8, ','], ["B'Day (2006)", 9, 'entity'], [',', 10, ','], ['which', 11, 'subject'], ['contained', 12, 'predicate'], ['hits', 13, 'object'], ['"', 14, '``'], ['Déjà Vu', 15, 'entity'], ['"', 16, "''"], [',', 17, ','], ['"Irreplaceable', 18, 'entity'], ['"', 19, "''"], [',', 20, ','], ['and', 21, 'CC'], ['"Beautiful Liar', 22, 'entity'], ['"', 23, "''"]], [['Beyoncé', 25, 'subject'], ['also', 26, 'RB'], ['ventured into', 27, 'predicate'], ['acting', 28, 'entity'], [',', 29, ','], ['with', 30, 'predicate'], ['a Golden Globe-nominated performance', 31, 'object'], ['in', 32, 'predicate'], ['Dreamgirls (2006)', 33, 'object'], [',', 34, ','], ['and', 35, 'CC'], ['starring', 36, 'predicate'], ['roles', 37, 'object'], ['in', 38, 'predicate'], ['The Pink Panther (2006)', 39, 'object'], ['and', 40, 'CC'], ['Obsessed (2009)', 41, 'entity']], [['Her marriage', 43, 'subject'], ['to rapper', 44, 'TO'], ['Jay Z', 45, 'object'], ['and', 46, 'CC'], ['portrayal', 47, 'NN'], ['of', 48, 'predicate'], ['Etta James', 49, 'object'], ['in', 50, 'predicate'], ['Cadillac Records (2008)', 51, 'object'], ['influenced', 52, 'predicate'], ['her third album', 53, 'object'], [',', 54, ','], ['I', 55, 'subject'], ['Am', 56, 'predicate'], ['...', 57, 'NFP'], ['Sasha Fierce (2008)', 58, 'entity'], [',', 59, ','], ['which', 60, 'subject'], ['saw', 61, 'predicate'], ['the birth of her alter', 62, 'object'], ['-', 63, 'HYPH'], ['ego Sasha Fierce', 64, 'entity'], ['and', 65, 'CC'], ['earned', 66, 'predicate'], ['a record-setting six Grammy Awards', 67, 'object'], ['in', 68, 'predicate'], ['2010', 69, 'object'], [',', 70, ','], ['including', 71, 'predicate'], ['Song of the Year', 72, 'object'], ['for', 73, 'predicate'], ['"Single Ladies (Put a Ring on It)', 74, 'entity'], ['"', 75, "''"]], [['Beyoncé', 77, 'subject'], ['took', 78, 'predicate'], ['a hiatus', 79, 'object'], ['from', 80, 'predicate'], ['music', 81, 'object'], ['in', 82, 'predicate'], ['2010', 83, 'object'], ['and', 84, 'CC'], ['took over', 85, 'predicate'], ['management of her career', 86, 'object'], [';', 87, ':'], ['her fourth album 4 (2011)', 88, 'subject'], ['was', 89, 'predicate'], ['subsequently', 90, 'RB'], ['mellower', 91, 'JJR'], ['in', 92, 'predicate'], ['tone', 93, 'object'], [',', 94, ','], ['exploring', 95, 'predicate'], ['1970s funk', 96, 'object'], [',', 97, ','], ['1980s pop', 98, 'entity'], [',', 99, ','], ['and', 100, 'CC'], ['1990s soul', 101, 'entity']], [['Her critically acclaimed fifth studio album', 103, 'subject'], [',', 104, ','], ['Beyoncé (2013)', 105, 'entity'], [',', 106, ','], ['was distinguished from', 107, 'predicate'], ['previous releases', 108, 'object'], ['by', 109, 'predicate'], ['its experimental production', 110, 'object'], ['and', 111, 'CC'], ['exploration of darker themes', 112, 'entity']]]
    relations = []
    #loose_entities = []
    for s in sentences:
        if len(s) == 0: continue
        preds = get_predicate(s) # Get all verbs 例如：{0: 'Following', 2: 'in', 6: 'released', 12: 'contained'}
        """
        if preds == {}: 
            preds = {p[1]:p[0] for p in s if (p[2] == 'JJ' or p[2] == 'IN' or p[2] == 'CC' or
                     p[2] == 'RP' or p[2] == ':' or p[2] == 'predicate' or
                     p[2] =='-LRB-' or p[2] =='-RRB-') }
            if preds == {}:
                #print('\npred = 0', s)
                preds = {p[1]:p[0] for p in s if (p[2] == ',')}
                if preds == {}:
                    ents = [e[0] for e in s if e[2] == 'entity']
                    if (ents):
                        loose_entities = ents # not significant for now
                        #print("Loose entities = ", ents)
        """
        if preds: # 如果谓词字典不为空， 例如{0: 'Following', 2: 'in', 6: 'released', 12: 'contained'}
            if (len(preds) == 1):
                #print("preds = ", preds)
                predicate = list(preds.values())[0]
                if (len(predicate) < 2):
                    predicate = 'is'
                #print(s)
                ents = [e[0] for e in s if e[2] == 'entity']
                #print('ents = ', ents)
                for i in range(1, len(ents)):
                    relations.append([ents[0], predicate, ents[i]])

            pred_ids = list(preds.keys())
            pred_ids.append(s[0][1])
            pred_ids.append(s[len(s)-1][1])
            pred_ids.sort()
                    
            for i in range(1, len(pred_ids)-1):
                predicate = preds[pred_ids[i]]
                adps_subjs = get_positions(s, pred_ids[i-1], pred_ids[i])
                subjs = get_subjects(s, pred_ids[i-1], pred_ids[i], adps_subjs)
                adps_objs = get_positions(s, pred_ids[i], pred_ids[i+1])
                objs = get_objects(s, pred_ids[i], pred_ids[i+1], adps_objs)
                for k_s, subj in subjs.items():                
                    for k_o, obj in objs.items():
                        obj_prev_id = int(k_o) - 1
                        if obj_prev_id in adps_objs: # at, in, of
                            relations.append([subj, predicate + ' ' + adps_objs[obj_prev_id], obj])
                        else:
                            relations.append([subj, predicate, obj]) # 提取可能的动词、谓词等，作为候选关系词，并构建成三元组，但其中包含很多指代词
    
    ### Read coreferences: coreference files are TAB separated values 上面暂时构建的三元组包含大量的指代词，需要根据前面的共指消解进行更新
    coreferences = []
    for val in corefs: # 遍历每个指代消解
        if val[0].strip() != val[1].strip():
            if len(val[0]) <= 50 and len(val[1]) <= 50:
                co_word = val[0]
                real_word = val[1].strip('[,- \'\n]*')
                real_word = re.sub("'s$", '', real_word, flags=re.UNICODE)
                if (co_word != real_word):
                    coreferences.append([co_word, real_word])
            else:
                co_word = val[0]
                real_word = ' '.join((val[1].strip('[,- \'\n]*')).split()[:7])
                real_word = re.sub("'s$", '', real_word, flags=re.UNICODE)
                if (co_word != real_word):
                    coreferences.append([co_word, real_word])
                
    # Resolve corefs 先对头实体进行消解，在基础上再对尾实体进行消解
    triples_object_coref_resolved = []
    triples_all_coref_resolved = []
    for s, p, o in relations: # 遍历每个三元组
        coref_resolved = False
        for co in coreferences: # 遍历每个指代
            if (s == co[0]): # 如果头实体和指代相同，则更新
                subj = co[1]
                triples_object_coref_resolved.append([subj, p, o])
                coref_resolved = True
                break
        if not coref_resolved: # 该三元组不存在指代问题，直接将原始的三元组加入
            triples_object_coref_resolved.append([s, p, o])

    for s, p, o in triples_object_coref_resolved: # 遍历每个第一次消解后的三元组
        coref_resolved = False
        for co in coreferences: # 遍历每个指代
            if (o == co[0]): # 如果尾实体和指代相同，则更新
                obj = co[1]
                triples_all_coref_resolved.append([s, p, obj])
                coref_resolved = True
                break
        if not coref_resolved:
            triples_all_coref_resolved.append([s, p, o])
    return(triples_all_coref_resolved) # 返回最终消解的三元组

### ==================================================================================================
## Get more using Network shortest_paths

def get_graph(triples):
    G = nx.DiGraph()
    for s, p, o in triples:
        G.add_edge(s, o, key=p)
    return G

def get_entities_with_capitals(G):
    entities = []
    for node in G.nodes():
        if (any(ch.isupper() for ch in list(node))):
            entities.append(node)
    return entities

def get_paths_between_capitalised_entities(triples):
    
    g = get_graph(triples)
    ents_capitals = get_entities_with_capitals(g)
    paths = []
    #print('\nShortest paths among capitalised words -------------------')
    for i in range(0, len(ents_capitals)):
        n1 = ents_capitals[i]
        for j in range(1, len(ents_capitals)):
            try:
                n2 = ents_capitals[j]
                path = nx.shortest_path(g, source=n1, target=n2)
                if path and len(path) > 2:
                    paths.append(path)
                path = nx.shortest_path(g, source=n2, target=n1)
                if path and len(path) > 2:
                    paths.append(path)
            except Exception:
                continue
    return g, paths

def get_paths(doc_triples):
    triples = []
    g, paths = get_paths_between_capitalised_entities(doc_triples) # 计算中介中心性、度、最短路径等任务。g为图对象，paths为所有的最短路径
    for p in paths:
        path = [(u, g[u][v]['key'], v) for (u, v) in zip(p[0:], p[1:])]
        length = len(p)
        if (path[length-2][1] == 'in' or path[length-2][1] == 'at' or path[length-2][1] == 'on'):
            if [path[0][0], path[length-2][1], path[length-2][2]] not in triples:
                triples.append([path[0][0], path[length-2][1], path[length-2][2]])
        elif (' in' in path[length-2][1] or ' at' in path[length-2][1] or ' on' in path[length-2][1]):
            if [path[0][0], path[length-2][1], path[length-2][2]] not in triples:
                triples.append([path[0][0], 'in', path[length-2][2]])
    for t in doc_triples:
        if t not in triples:
            triples.append(t)
    return triples

def get_center(nodes):
    center = ''
    if (len(nodes) == 1):
        center = nodes[0]
    else:   
        # Capital letters and longer is preferred
        cap_ents = [e for e in nodes if any(x.isupper() for x in e)]
        if (cap_ents):
            center = max(cap_ents, key=len)
        else:
            center = max(nodes, key=len)
    return center

def connect_graphs(mytriples):
    G = nx.DiGraph()
    for s, p, o in mytriples:
        G.add_edge(s, o, p=p)        
    
    """
    # Get components
    graphs = list(nx.connected_component_subgraphs(G.to_undirected()))
    
    # Get the largest component
    largest_g = max(graphs, key=len)
    largest_graph_center = ''
    largest_graph_center = get_center(nx.center(largest_g))
    
    # for each graph, find the centre node
    smaller_graph_centers = []
    for g in graphs:        
        center = get_center(nx.center(g))
        smaller_graph_centers.append(center)

    for n in smaller_graph_centers:
        if (largest_graph_center is not n):
            G.add_edge(largest_graph_center, n, p='with')
    """
    return G
        
def rank_by_degree(mytriples): #, limit):
    G = connect_graphs(mytriples)
    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')
    
    # Use this to draw the graph
    #draw_graph_centrality(G, degree_dict)

    Egos = nx.DiGraph()
    for a, data in sorted(G.nodes(data=True), key=lambda x: x[1]['degree'], reverse=True):
        ego = nx.ego_graph(G, a)
        Egos.add_edges_from(ego.edges(data=True))
        Egos.add_nodes_from(ego.nodes(data=True))
        
        #if (nx.number_of_edges(Egos) > 20):
        #    break
       
    ranked_triples = []
    for u, v, d in Egos.edges(data=True):
        ranked_triples.append([u, d['p'], v])
    return ranked_triples

# 抽取三元组
def extract_triples(text):
    df_tagged, corefs, tokens = tagger(text) # pipeline处理文本，并返回每个token的特征，以及共指消解的结果
    doc_triples = create_triples(df_tagged, corefs) # 获得可能的关系词，并构建三元组，并进行指代消解，返回所有三元组（发现有消解过的，还有没有消解过的）
    all_triples = get_paths(doc_triples) # 例子：[['Beyoncé', 'in', 'Dreamgirls (2006)'], ['Beyoncé', 'in', 'The Pink Panther (2006)'], ['Beyoncé', 'in', 'Obsessed (2009)'], ['a Golden Globe-nominated performance', 'in', 'The Pink Panther (2006)'], ['a Golden Globe-nominated performance', 'in', 'Obsessed (2009)'], ['Dreamgirls (2006)', 'in', 'The Pink Panther (2006)'], ['Dreamgirls (2006)', 'in', 'Obsessed (2009)'], ['Her marriage', 'in', 'Cadillac Records (2008)'], ['Jay Z', 'in', 'Cadillac Records (2008)'], ["the disbandment of Destiny's Child", 'in', 'June 2005'], ["the disbandment of Destiny's Child", 'in', 'she'], ['June 2005', 'released', 'her second solo album'], ['June 2005', 'released', "B'Day (2006)"], ['June 2005', 'released', 'which'], ['she', 'released', 'her second solo album'], ['she', 'released', "B'Day (2006)"], ['she', 'released', 'which'], ['her second solo album', 'contained', 'hits'], ['her second solo album', 'contained', 'Déjà Vu'], ['her second solo album', 'contained', '"Irreplaceable'], ['her second solo album', 'contained', '"Beautiful Liar'], ["B'Day (2006)", 'contained', 'hits'], ["B'Day (2006)", 'contained', 'Déjà Vu'], ["B'Day (2006)", 'contained', '"Irreplaceable'], ["B'Day (2006)", 'contained', '"Beautiful Liar'], ['which', 'contained', 'hits'], ['which', 'contained', 'Déjà Vu'], ['which', 'contained', '"Irreplaceable'], ['which', 'contained', '"Beautiful Liar'], ['Beyoncé', 'ventured into', 'acting'], ['acting', 'with', 'a Golden Globe-nominated performance'], ['a Golden Globe-nominated performance', 'in', 'Dreamgirls (2006)'], ['Dreamgirls (2006)', 'starring', 'roles'], ['roles', 'in', 'The Pink Panther (2006)'], ['roles', 'in', 'Obsessed (2009)'], ['Her marriage', 'of', 'Etta James'], ['Jay Z', 'of', 'Etta James'], ['Etta James', 'in', 'Cadillac Records (2008)'], ['Cadillac Records (2008)', 'influenced', 'her third album'], ['Cadillac Records (2008)', 'influenced', 'I'], ['her third album', 'Am', 'Sasha Fierce (2008)'], ['her third album', 'Am', 'which'], ['I', 'Am', 'Sasha Fierce (2008)'], ['I', 'Am', 'which'], ['Sasha Fierce (2008)', 'saw', 'the birth of her alter'], ['Sasha Fierce (2008)', 'saw', 'ego Sasha Fierce'], ['which', 'saw', 'the birth of her alter'], ['which', 'saw', 'ego Sasha Fierce'], ['the birth of her alter', 'earned', 'a record-setting six Grammy Awards'], ['ego Sasha Fierce', 'earned', 'a record-setting six Grammy Awards'], ['a record-setting six Grammy Awards', 'in', '2010'], ['2010', 'including', 'Song of the Year'], ['Song of the Year', 'for', '"Single Ladies (Put a Ring on It)'], ['Beyoncé', 'took', 'a hiatus'], ['a hiatus', 'from', 'music'], ['music', 'in', '2010'], ['2010', 'took over', 'management of her career'], ['2010', 'took over', 'her fourth album 4 (2011)'], ['tone', 'exploring', '1970s funk'], ['tone', 'exploring', '1980s pop'], ['tone', 'exploring', '1990s soul'], ['Her critically acclaimed fifth studio album', 'was distinguished from', 'previous releases'], ['Beyoncé (2013)', 'was distinguished from', 'previous releases'], ['previous releases', 'by', 'its experimental production'], ['previous releases', 'by', 'exploration of darker themes']]
    filtered_triples = []    
    for s, p, o in all_triples: # 过滤掉一些不符合或多余的三元组（包含停用词，指代类词等三元组删除）
        if ([s, p, o] not in filtered_triples):
            if s.lower() in all_stop_words or o.lower() in all_stop_words:
                continue
            elif s == p:
                continue
            if s.isdigit() or o.isdigit():
                continue
            if '%' in o or '%' in s: #= 11.96
                continue
            if (len(s) < 2) or (len(o) < 2):
                continue
            if (s.islower() and len(s) < 4) or (o.islower() and len(o) < 4):
                continue
            if s == o:
                continue            
            subj = s.strip('[,- :\'\"\n]*')
            pred = p.strip('[- :\'\"\n]*.')
            obj = o.strip('[,- :\'\"\n]*')
            
            for sw in ['a', 'an', 'the', 'its', 'their', 'his', 'her', 'our', 'all', 'old', 'new', 'latest', 'who', 'that', 'this', 'these', 'those']:
                subj = ' '.join(word for word in subj.split() if not word == sw)
                obj = ' '.join(word for word in obj.split()  if not word == sw)
            subj = re.sub("\s\s+", " ", subj)
            obj = re.sub("\s\s+", " ", obj)
            
            if subj and pred and obj:
                filtered_triples.append([subj, pred, obj])

    #TRIPLES = rank_by_degree(filtered_triples)
    return filtered_triples, df_tagged, tokens, corefs

def draw_graph_centrality(G, dictionary):
    # plt.figure(figsize=(12,10))
    # pos = nx.spring_layout(G)
    # #print("Nodes\n", G.nodes(True))
    # #print("Edges\n", G.edges())
    
    # nx.draw_networkx_nodes(G, pos, 
    #         nodelist=dictionary.keys(),
    #         with_labels=False,
    #         edge_color='black',
    #         width=1,
    #         linewidths=1,
    #         node_size = [v * 150 for v in dictionary.values()],
    #         node_color='blue',
    #         alpha=0.5)
    # edge_labels = {(u, v): d["p"] for u, v, d in G.edges(data=True)}
    # #print(edge_labels)
    # nx.draw_networkx_edge_labels(G, pos,
    #                        font_size=10,
    #                        edge_labels=edge_labels,
    #                        font_color='blue')
    # nx.draw(G, pos, with_labels=True, node_size=1, node_color='blue')
    pass
    
if __name__ == "__main__":
    """
    Celebrity chef Jamie Oliver's British restaurant chain has become insolvent, putting 1,300 jobs at risk. The firm said Tuesday that it had gone into administration, a form of bankruptcy protection, and appointed KPMG to oversee the process.The company operates 23 Jamie's Italian restaurants in the U.K. The company had been seeking buyers amid increased competition from casual dining rivals, according to The Guardian. Oliver began his restaurant empire in 2002 when he opened Fifteen in London. Oliver, known around the world for his cookbooks and television shows, said he was "deeply saddened by this outcome and would like to thank all of the staff and our suppliers who have put their hearts and souls into this business for over a decade. "He said "I appreciate how difficult this is for everyone affected." I’m devastated that our much-loved UK restaurants have gone into administration.
    """
    """BYD debuted its E-SEED GT concept car and Song Pro SUV alongside its all-new e-series models at the Shanghai International Automobile Industry Exhibition. The company also showcased its latest Dynasty series of vehicles, which were recently unveiled at the company’s spring product launch in Beijing."""
    text = """
    BYD debuted its E-SEED GT concept car and Song Pro SUV alongside its all-new e-series models at the Shanghai International Automobile Industry Exhibition. The company also showcased its latest Dynasty series of vehicles, which were recently unveiled at the company’s spring product launch in Beijing. A total of 23 new car models were exhibited at the event, held at Shanghai’s National Convention and Exhibition Center, fully demonstrating the BYD New Architecture (BNA) design, the 3rd generation of Dual Mode technology, plus the e-platform framework. Today, China’s new energy vehicles have entered the ‘fast lane’, ushering in an even larger market outbreak. Presently, we stand at the intersection of old and new kinetic energy conversion for mobility, but also a new starting point for high-quality development. To meet the arrival of complete electrification, BYD has formulated a series of strategies, and is well prepared.
    """
    """
    An arson fire caused an estimated $50,000 damage at a house on Mt. Soledad that was being renovated, authorities said Friday.San Diego police were looking for the arsonist, described as a Latino man who was wearing a red hat, blue shirt and brown pants, and may have driven away in a small, black four-door car.A resident on Palomino Court, off Soledad Mountain Road, called 9-1-1 about 9:45 a.m. to report the house next door on fire, with black smoke coming out of the roof, police said. Firefighters had the flames knocked down 20 minutes later, holding the damage to the attic and roof, said City spokesperson Alec Phillip. No one was injured.Metro Arson Strike Team investigators were called and they determined the blaze had been set intentionally, Phillip said.Police said one or more witnesses saw the suspect run south from the house and possibly leave in the black car.
    """
    mytriples = extract_triples(text)
    
    print('\n\nFINAL TRIPLES = ', len(mytriples))
    for t in mytriples:
        print(t)