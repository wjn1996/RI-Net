import difflib
import Levenshtein
import networkx as nx
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt



def get_equal_rate(str1, str2):
    score = Levenshtein.ratio(str1, str2)
    if score > 0.80:
        return True
    else:
        return False



# def get_equal_rate(str1, str2):
#     score = difflib.SequenceMatcher(None, str1, str2).quick_ratio()
#     if score > 0.85:
#         return True
#     else:
#         return False
def int_list_to_str(s):
    return '_'.join(str(i) for i in s)

def str_to_int_list(s):
    return [int(i) for i in s.split('_') if i != '']



# # 找到一条从start到end的路径
# def findPath(graph, start, end, path=[]):
#     path = path + [start]
#     if start == end:
#         return path
#     for node in graph[start]:
#         if node not in path:
#             newpath = findPath(graph, node, end, path)
#             if newpath:
#                 return newpath
#     return []
#
#
# # 找到所有从start到end的路径
# def findAllPath(graph, start, end, path=[]):
#     path = path + [start]
#     if start == end:
#         return [path]
#
#     paths = []  # 存储所有路径
#     for node in graph[start]:
#         if node not in path:
#             newpaths = findAllPath(graph, node, end, path)
#             for newpath in newpaths:
#                 paths.append(newpath)
#     return paths
#
#
# # 查找最短路径
# def findShortestPath(graph, start, end, path=[]):
#     path = path + [start]
#     if start == end:
#         return path
#
#     shortestPath = []
#     for node in graph[start]:
#         if node not in path:
#             newpath = findShortestPath(graph, node, end, path)
#             if newpath:
#                 if not shortestPath or len(newpath) < len(shortestPath):
#                     shortestPath = newpath
#     return shortestPath

def find_Rel_Path(ent_path, entpair2tripleid, refine_triples_with_cls_ids):
    # 根据实体路径，两两获取对应的关系边
    rel_path = []
    for i in range(len(ent_path) - 1):
        head = ent_path[i]
        tail = ent_path[i+1]
        ent_pair = (head, tail)
        if ent_pair not in entpair2tripleid.keys():
            ent_pair = (tail, head)
        rel_path.append(refine_triples_with_cls_ids[entpair2tripleid[ent_pair]][1])
    return rel_path

def find_Shortest_Path(refine_triples_ids, refine_triples_with_cls_ids, entity_ids, entpair2tripleid, cls_id):
    # graph: [[head, rel, tail], ...]
    # cls_str = int_list_to_str([cls_id])
    path_dicts = []
    for batch_i in range(len(refine_triples_ids)): # 遍历每个batch样本
        ent_pair = [(int_list_to_str(head), int_list_to_str(tail)) for head, _, tail in refine_triples_ids[batch_i][0]]
        # ent_with_cls_pair = [(int_list_to_str(head), int_list_to_str(tail)) for head, _, tail in refine_triples_with_cls_ids]
        G1 = nx.Graph()
        G1.add_edges_from(ent_pair)
        # G2.add_edges_from(ent_with_cls_pair)
        nx.draw(G1, with_labels=True)
        # nx.draw(G2, with_labels=True)
        entity_list = [i for i in entity_ids[batch_i][0].keys()]
        # 当两个实体不连通时，再通过额外添加的CLS结点计算路径
        path_dict = dict()
        for i in range(len(entity_list)):
            for j in range(i, len(entity_list)):
                if i == j:
                    ent_path = [str_to_int_list(entity_list[i]), str_to_int_list(entity_list[i])]
                    rel_path = [[cls_id]]
                elif nx.has_path(G1, entity_list[i], entity_list[j]):
                    ent_path = nx.shortest_path(G1, source=entity_list[i], target=entity_list[j])
                    rel_path = find_Rel_Path(ent_path, entpair2tripleid[batch_i][0], refine_triples_with_cls_ids[batch_i][0])
                    ent_path = [str_to_int_list(k) for k in ent_path]
                else:
                    ent_path = [str_to_int_list(entity_list[i]), [cls_id], str_to_int_list(entity_list[j])]
                    rel_path = [[cls_id], [cls_id]]
                path_dict[(entity_list[i], entity_list[j])] = [ent_path, rel_path]
        path_dicts.append(path_dict)
    return path_dicts


if __name__ == '__main__':
    get_equal_rate('Beyoncé', 'Beyonce')

    # graph = {'A': ['B', 'C', 'D'],
    #          'B': ['E'],
    #          'C': ['D', 'F'],
    #          'D': ['B', 'E', 'G'],
    #          'E': [],
    #          'F': ['D', 'G'],
    #          'G': ['E']}
    #
    # onepath = findPath(graph, 'A', 'G')
    # print('一条路径:', onepath)
    #
    # allpath = findAllPath(graph, 'A', 'G')
    # print('\n所有路径：', allpath)
    #
    # shortpath = findShortestPath(graph, 'A', 'G')
    # print('\n最短路径：', shortpath)