from pathlib import Path

from graph.tf_idf import TFIDF

import numpy as np

from sitter.kast2core import KASTParse


class MatrixConstruction:
    def __init__(self, sr_class):
        self.Wssm = 1
        self.Wcdm = 1
        self.Wcsm = 1
        self.method_list = sr_class.method_list
        self.matrix = np.zeros((len(self.method_list), len(self.method_list)))
        self.field_name_list = [o.field_name for o in sr_class.field_list]
        self.num_method = len(self.method_list)

    def generate_matrix(self):
        self.calculate_SSM()
        # print(self.matrix)
        self.calculate_CDM()
        # print(self.matrix)
        self.calculate_CSM()
        # print(self.matrix)

    def get_all_matrix(self):
        self.calculate_SSM()
        ssm = self.matrix
        self.matrix = np.zeros((len(self.method_list), len(self.method_list)))

        self.calculate_bi_CDM()
        cdm = self.matrix
        self.matrix = np.zeros((len(self.method_list), len(self.method_list)))

        self.calculate_CSM()
        csm = self.matrix
        return ssm, cdm, csm


    def calculate_bi_CDM(self):
        method_calls = []
        method_name_list = [o.method_name for o in self.method_list]
        for sr_method in self.method_list:
            mmu = sr_method.get_all_method_used(method_name_list)
            method_calls.extend(mmu)

        for i in range(0, self.num_method):
            for j in range(i, self.num_method):
                if i == j:
                    self.matrix[i][j] += 1.0 * self.Wcdm
                    continue

                cdm_ij = self.get_frequency(self.method_list[j].method_name, method_calls)
                if cdm_ij != 0:
                    mi_name_list = self.method_list[i].get_all_method_used(method_name_list)
                    cdm_ij = self.get_frequency(self.method_list[j].method_name, mi_name_list) / cdm_ij,

                cdm_ji = self.get_frequency(self.method_list[i].method_name, method_calls)
                if cdm_ji != 0:
                    mj_name_list = self.method_list[j].get_all_method_used(method_name_list)
                    cdm_ji = self.get_frequency(self.method_list[i].method_name, mj_name_list) / cdm_ji

                if type(cdm_ij) is tuple:
                    cdm_ij = cdm_ij[0]
                if type(cdm_ji) is tuple:
                    cdm_ji = cdm_ji[0]

                self.matrix[i][j] += cdm_ij
                self.matrix[j][i] += cdm_ji

    def calculate_SSM(self):
        for i in range(0, self.num_method):
            for j in range(i, self.num_method):
                union = []
                all_f_i = self.method_list[i].get_all_fields(self.field_name_list)
                all_f_i = list(set(all_f_i))
                union.extend(all_f_i)

                all_f_j = self.method_list[j].get_all_fields(self.field_name_list)
                all_f_j = list(set(all_f_j))
                union.extend(all_f_j)

                if i == j:
                    self.matrix[i][j] += 1.0 * self.Wssm
                    continue

                union = list(set(union))
                ssm_ij = len(union)
                if ssm_ij == 0:
                    continue

                intersection = [o for o in all_f_i if o in all_f_j]

                ssm_ij = (len(intersection) / ssm_ij) * self.Wssm
                self.matrix[i][j] += ssm_ij
                self.matrix[j][i] += ssm_ij

    def get_frequency(self, method_name, method_name_list):
        count = 0
        for m in method_name_list:
            if m == method_name:
                count += 1
        return count

    def calculate_CDM(self):
        method_calls = []
        method_name_list = [o.method_name for o in self.method_list]
        for sr_method in self.method_list:
            mmu = sr_method.get_all_method_used(method_name_list)
            method_calls.extend(mmu)

        for i in range(0, self.num_method):
            for j in range(i, self.num_method):
                if i == j:
                    self.matrix[i][j] += 1.0 * self.Wcdm
                    continue

                cdm_ij = self.get_frequency(self.method_list[j].method_name, method_calls)
                if cdm_ij != 0:
                    mi_name_list = self.method_list[i].get_all_method_used(method_name_list)
                    cdm_ij = self.get_frequency(self.method_list[j].method_name, mi_name_list) / cdm_ij,

                cdm_ji = self.get_frequency(self.method_list[i].method_name, method_calls)
                if cdm_ji != 0:
                    mj_name_list = self.method_list[j].get_all_method_used(method_name_list)
                    cdm_ji = self.get_frequency(self.method_list[i].method_name, mj_name_list) / cdm_ji

                if type(cdm_ij) is tuple:
                    cdm_ij = cdm_ij[0]
                if type(cdm_ji) is tuple:
                    cdm_ji = cdm_ji[0]

                cdm_ij = max(cdm_ji, cdm_ij) * self.Wcdm
                self.matrix[i][j] += cdm_ij
                self.matrix[j][i] += cdm_ij

    def calculate_CSM(self):
        tf_idf = TFIDF()
        tf_idf.calc_with_statements(self.method_list)
        tf_idf_vectors = tf_idf.tfIdf_vectors
        euclidean_norm = tf_idf.euclidean_norm
        # print(self.num_method)
        for i in range(0, self.num_method):
            for j in range(i, self.num_method):
                if i == j:
                    self.matrix[i][j] += 1.0 * self.Wcsm
                    continue

                if euclidean_norm[i] != 0 and euclidean_norm[j] != 0:
                    csm_ij = self.vec_product(tf_idf_vectors[i], tf_idf_vectors[j]) / (euclidean_norm[i] * euclidean_norm[j])
                    csm_ij *= self.Wcsm
                    self.matrix[i][j] += csm_ij
                    self.matrix[j][i] += csm_ij


    def calculate_CSM_W2V(self, w2v):
        matrix = np.zeros((len(self.method_list), len(self.method_list)))
        tf_idf = TFIDF()
        tf_idf.calc_with_statements(self.method_list)
        tf_idf_vectors = tf_idf.tfIdf_vectors
        euclidean_norm = tf_idf.euclidean_norm
        processed_method_list = tf_idf.processed_method_list
        # print(self.num_method)
        for i in range(0, self.num_method):
            for j in range(i, self.num_method):
                if i == j:
                    matrix[i][j] += 1.0 * self.Wcsm
                    continue

                if euclidean_norm[i] != 0 and euclidean_norm[j] != 0:
                    # csm_ij = self.vec_product(tf_idf_vectors[i], tf_idf_vectors[j]) / (euclidean_norm[i] * euclidean_norm[j])

                    processed_method_list[i] = list(set(processed_method_list[i]))
                    processed_method_list[j] = list(set(processed_method_list[j]))
                    str_i = " ".join(processed_method_list[i])
                    str_j = " ".join(processed_method_list[j])
                    csm_ij = w2v.get_word_sim_score(str_i, str_j)

                    csm_ij *= self.Wcsm
                    matrix[i][j] += csm_ij
                    matrix[j][i] += csm_ij

        return matrix

    def calculate_CSM_doc_sim(self, doc_sim):
        matrix = np.zeros((len(self.method_list), len(self.method_list)))
        tf_idf = TFIDF()
        tf_idf.calc_with_statements(self.method_list)
        tf_idf_vectors = tf_idf.tfIdf_vectors
        euclidean_norm = tf_idf.euclidean_norm
        processed_method_list = tf_idf.processed_method_list
        # print(self.num_method)
        for i in range(0, self.num_method):
            for j in range(i, self.num_method):
                if i == j:
                    matrix[i][j] += 1.0 * self.Wcsm
                    continue

                if euclidean_norm[i] != 0 and euclidean_norm[j] != 0:
                    # csm_ij = self.vec_product(tf_idf_vectors[i], tf_idf_vectors[j]) / (euclidean_norm[i] * euclidean_norm[j])

                    processed_method_list[i] = list(set(processed_method_list[i]))
                    processed_method_list[j] = list(set(processed_method_list[j]))
                    str_i = " ".join(processed_method_list[i])
                    str_j = " ".join(processed_method_list[j])
                    csm_ij = doc_sim.calculate_similarity(str_i, str_j)

                    csm_ij *= self.Wcsm
                    matrix[i][j] += csm_ij
                    matrix[j][i] += csm_ij

        return matrix

    def vec_product(self, vector1, vector2):
        prod = 0
        for i in range(0, len(vector1)):
            prod += vector1[i]*vector2[i]
        return prod

    # def calculate_CSM(self):

def hac_metric(matrix):
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.cluster import AgglomerativeClustering
    import scipy.cluster.hierarchy as sch

    X = matrix
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()

    model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    model.fit(X)
    labels = model.labels_
    plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s=50, marker='o', color='red')
    plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s=50, marker='o', color='blue')
    plt.show()

def test_hac():

    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.cluster import AgglomerativeClustering
    import scipy.cluster.hierarchy as sch

    dataset = pd.read_csv('/Users/zhang.hanyu/Documents/workspace/research/research/Snow/dataset_generation/lc/Mall_Customers.csv')
    X = dataset.iloc[:, [3, 4]].values
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()

    # model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    # model.fit(X)
    # labels = model.labels_
    # plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s=50, marker='o', color='red')
    # plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s=50, marker='o', color='blue')
    # plt.scatter(X[labels == 2, 0], X[labels == 2, 1], s=50, marker='o', color='green')
    # plt.scatter(X[labels == 3, 0], X[labels == 3, 1], s=50, marker='o', color='purple')
    # plt.scatter(X[labels == 4, 0], X[labels == 4, 1], s=50, marker='o', color='orange')
    # plt.show()

if __name__ == '__main__':
    project_path = Path("/Users/zhang.hanyu/Documents/workspace/research/research/Snow/tmp/so")
    ast = KASTParse(project_path, "java")
    ast.setup()
    sr_project = ast.do_parse()
    mc = 0
    for program in sr_project.program_list:
        for cls in program.class_list:
            mc = MatrixConstruction(cls)
            mc.generate_matrix()
            matrix = mc.matrix
            print(matrix)
            # hac_metric(matrix)
    # test_hac()