from graph.stop_word_remover import StopWordRemover
from reflect.sr_statement import SRIFStatement, SRTRYStatement, SRWhileStatement, SRFORStatement, SRSwitchStatement


class MetricsCalculator:

    def __init__(self, sr_class):
        self.sr_class = sr_class

    def get_method_loc(self, sr_method):
        result = 0
        result = sr_method.end_line - sr_method.start_line + 1
        return result

    def get_method_cc(self, sr_method):
        result = 0
        total_statement = sr_method.get_all_statement(exclude_special=False)
        condition_nodes = []
        for st in total_statement:
            if type(st) == SRIFStatement:
                condition_nodes.append(st)
        result = len(condition_nodes) + 1
        return result

    def get_method_pc(self, sr_method):
        result = 0
        result = len(sr_method.param_list)
        return result

    def statement_special_key_filter(self, word_list):
        new_st_l = []
        method_name_l = []
        var_name_l = []
        java_keywords = ["boolean", "int", "long", "short", "byte", "float", "double", "char", "class", "interface",
                         "if", "else", "do", "while", "for", "switch", "case", "default", "break", "continue", "return",
                         "try", "catch", "finally", "public", "protected", "private", "final", "void", "static",
                         "strict", "abstract", "transient", "synchronized", "volatile", "native", "package", "import",
                         "throw", "throws", "extends", "implements", "this", "supper", "instanceof", "new", "true",
                         "false", "null", "goto", "const", "=", "*=", "/=", "%=", "+=", "-=", "<<=", ">>=", "&=", "!=", "^=", ">>>=", "++", "--", "=="]
        special_key = "[\n`~!@#$%^&*()+=\\-_|{}':;',\\[\\].<>/?~！@#￥%……&*（）——+|{}【】‘；：”“’。， 、？]"
        for i, w in enumerate(word_list):
            if w not in java_keywords and w not in special_key and not str(w).isdigit():
                if i < (len(word_list)-1) and word_list[i+1] == "(":
                    method_name_l.append(w)
                else:
                    var_name_l.append(w)
        return method_name_l, var_name_l

    def get_method_LCOM1(self, sr_method):
        fstl = []
        P=0
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            for j in range(i+1, len(fstl)):
                if self.check_common_word(
                    l1=list(fstl[i]["var_name_l"]),
                    l2=list(fstl[j]["var_name_l"]),
                ) is False:
                    P += 1


        # print(fstl)
        # print(P)
        return P

    def check_common_word(self, l1, l2):
        if len(l1) > len(l2):
            for w in l1:
                if w in l2:
                    return True
        else:
            for w in l2:
                if w in l1:
                    return True
        return False

    def get_method_LCOM2(self, sr_method):
        fstl = []
        P = 0
        Q = 0
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            for j in range(i + 1, len(fstl)):
                if self.check_common_word(
                        l1=list(fstl[i]["var_name_l"]),
                        l2=list(fstl[j]["var_name_l"]),
                ) is False:
                    P += 1
                else:
                    Q += 1

        # print(Q)
        # print(P)
        lcom2 = P - Q

        if lcom2 < 0:
            lcom2 = 0
        return lcom2

    def get_method_LCOM3(self, sr_method):
        fstl = []
        compnent_count = []
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            for j in range(i + 1, len(fstl)):
                if self.check_common_word(
                        l1=list(fstl[i]["var_name_l"]),
                        l2=list(fstl[j]["var_name_l"]),
                ) is True:
                    if i not in compnent_count:
                        compnent_count.append(i)
                    if j not in compnent_count:
                        compnent_count.append(j)
        return len(compnent_count)

    def get_method_LCOM4(self, sr_method):
        fstl = []
        method_call_count = []
        component_count = 0
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            if len(fstl[i]["method_name_l"]) > 0:
                component_count += 1
                for m in fstl[i]["method_name_l"]:
                    if m not in method_call_count:
                        method_call_count.append(m)
                        component_count+=1


        return component_count

    def camel_case_split(self, str):
        if len(str) == 0:
            return ""
        words = [[str[0]]]

        for c in str[1:]:
            if words[-1][-1].islower() and c.isupper():
                words.append(list(c))
            else:
                words[-1].append(c)

        return [''.join(word) for word in words]

    def get_tsmc(self, sr_method, doc_sim):
        method_name_txt = self.camel_case_split(sr_method.method_name)
        method_name_txt = " ".join(method_name_txt)

        class_name_txt = self.camel_case_split(self.sr_class.class_name)
        class_name_txt = " ".join(class_name_txt)

        source_doc = method_name_txt
        target_docs = class_name_txt
        sim_score = doc_sim.calculate_similarity(source_doc, target_docs)
        sim_score = round(sim_score, 2)
        return sim_score

    def get_method_nfdi(self, sr_method, class_list):
        result = 0
        foreign_field_name_list = []
        foreign_method_name_list = []

        local_field_name_list = []
        local_method_name_list = []

        for cls in class_list:
            if cls.class_name == self.sr_class.class_name:
                continue

            for field in cls.field_list:
                foreign_field_name_list.append(field.field_name)
            for method in cls.method_list:
                foreign_method_name_list.append(method.method_name)

        for field in self.sr_class.field_list:
            local_field_name_list.append(field.field_name)

        for method in self.sr_class.method_list:
            local_method_name_list.append(method.method_name)

        for statement in sr_method.statement_list:
            for word in statement.to_node_word_list():
                if word in foreign_method_name_list or word in foreign_field_name_list:
                    if word not in local_method_name_list or word not in local_field_name_list:
                        result +=1
        return result

    def get_method_nldi(self, sr_method):
        result = 0
        local_field_name_list = []
        local_method_name_list = []

        for field in self.sr_class.field_list:
            local_field_name_list.append(field.field_name)

        for method in self.sr_class.method_list:
            local_method_name_list.append(method.method_name)

        for statement in sr_method.statement_list:
            for word in statement.to_node_word_list():
                if word in local_method_name_list or word in local_field_name_list:
                    result += 1
        return result

    def get_statement_abcl(self, sr_statement):
        result = 0
        a_feat = ["=", "*=", "/=", "%=", "+=", "-=", "<<=", ">>=", "&=", "!=", "^=", ">>>=", "++", "--"]

        if type(sr_statement) == SRIFStatement or type(sr_statement) == SRTRYStatement:
            result = 3
        elif type(sr_statement) == SRFORStatement or type(sr_statement) == SRWhileStatement:
            result = 4
        else:
            if "(" in sr_statement.word_list or "new" in sr_statement.word_list:
                result = 2
            else:
                for word in sr_statement.word_list:
                    if word in a_feat:
                        result = 1
                        break
        return result

    def get_statement_fuc(self, sr_statement):
        result = 0
        field_name_list = []
        # if len(self.field_name_list) == 0:
        for field in self.sr_class.field_list:
            field_name_list.append(field.field_name)
        for word in sr_statement.to_node_word_list():
            if word in field_name_list:
                result += 1
        return result

    def get_statement_lmuc(self, sr_statement):
        result = 0
        method_name_list = []

        for m in self.sr_class.method_list:
            method_name_list.append(m.method_name)
        for word in sr_statement.to_node_word_list():
            if word in method_name_list:
                result += 1
        return result

    def get_statement_vuc(self, sr_statement):
        method_name_l, var_name_l = self.statement_special_key_filter(sr_statement.to_node_word_list())
        return len(var_name_l)

    def get_statement_puc(self, sr_statement, param_name_list):
        puc = 0
        method_name_l, var_name_l = self.statement_special_key_filter(sr_statement.to_node_word_list())
        for var in var_name_l:
            if var in param_name_list:
                puc+=1
        return puc

    def get_statement_block_depth(self, sr_method, sr_statement):
        if sr_statement.block_depth == -1:
            self.calculate_depth(sr_method=sr_method)
            return sr_statement.block_depth
        else:
            return sr_statement.block_depth

    def calculate_depth(self, sr_method):
        self.__calculate_depth(statement_list=sr_method.statement_list, current_depth=0)

    def __calculate_depth(self, statement_list, current_depth):
        depth = current_depth
        for st in statement_list:
            st.block_depth = depth
            if type(st) == SRIFStatement:
                self.__calculate_depth(statement_list=st.pos_statement_list, current_depth=depth+1)
                self.__calculate_depth(statement_list=st.neg_statement_list, current_depth=depth+1)
            elif type(st) == SRFORStatement:
                self.__calculate_depth(statement_list=st.child_statement_list, current_depth=depth+1)
            elif type(st) == SRWhileStatement:
                self.__calculate_depth(statement_list=st.child_statement_list, current_depth=depth + 1)
            elif type(st) == SRTRYStatement:
                self.__calculate_depth(statement_list=st.try_statement_list, current_depth=depth + 1)
                for cb in st.catch_block_list:
                    self.__calculate_depth(statement_list=cb.child_statement_list, current_depth=depth + 1)
                self.__calculate_depth(statement_list=st.final_block_statement_list, current_depth=depth + 1)
            elif type(st) == SRSwitchStatement:
                for sc in st.switch_case_list:
                    self.__calculate_depth(statement_list=sc.statement_list, current_depth=depth + 1)

    def get_statement_wc(self, sr_statement):
        node_word_list = sr_statement.to_node_word_list()
        return len(node_word_list)

    def get_tsmm(self, sr_method, sr_statement, doc_sim):
        method_name_txt = self.camel_case_split(sr_method.method_name)
        method_name_txt = " ".join(method_name_txt)
        stop_word_remover = StopWordRemover()
        statement_txt = stop_word_remover.filter_statement_text(sr_statement)
        # statement_txt = " ".join(class_name_txt)

        source_doc = statement_txt
        target_docs = method_name_txt

        sim_score = doc_sim.calculate_similarity(source_doc, target_docs)
        sim_score = round(sim_score, 2)
        return sim_score

    def get_nom(self):
        method_count = 0
        method_count += len(self.sr_class.method_list)
        return method_count

    def get_cis(self):
        p_method_count = 0

        for method in self.sr_class.method_list:
            if len(method.modifiers) > 0:
                if method.modifiers[0] == "public":
                    p_method_count += 1
        return p_method_count

    def get_noa(self):
        field_count = 0

        field_count += len(self.sr_class.field_list)
        return field_count

    def get_nopa(self):
        p_field_count = 0

        for f in self.sr_class.field_list:
            if len(f.modifiers) > 0:
                if "public" in f.modifiers:
                    p_field_count += 1
        return p_field_count

    def get_atfd(self, class_list):
        atfd_count = 0
        current_field_n_l = []
        all_field_dic = {}
        for cls in class_list:
            all_field_n_l = []
            for f in cls.field_list:
                if "public" in f.modifiers and "static" in f. modifiers:
                    all_field_n_l.append(f.field_name)
            all_field_dic[cls.class_name] = all_field_n_l

        print(atfd_count)
        checked_list = []
        for m in self.sr_class.method_list:
            for st in m.statement_list:
                for index, word in enumerate(st.word_list):
                    if word in all_field_dic.keys():
                        if (index + 2) < len(st.word_list):
                            if st.word_list[index+2] in all_field_dic[word]:
                                if word != self.sr_class.class_name:
                                    ck = word + "." + st.word_list[index+2]
                                    if ck in checked_list:
                                        continue
                                    else:
                                        atfd_count += 1
                                        checked_list.append(ck)
        return atfd_count

    def get_wmc(self):
        wmc = 0

        for method in self.sr_class.method_list:
            cc = self.get_method_cc(method)
            wmc += cc
        return wmc

    def get_intersection_il(self, method1, method2, i_l):
        result = []
        i_l_1 = []
        i_l_2 = []

        i_l_n = [o.field_name for o in i_l]
        for statement in method1.get_all_statement(exclude_special=False):
            for word in statement.word_list:
                if word in i_l_n:
                    i_l_1.append(word)

        for statement in method2.get_all_statement(exclude_special=False):
            for word in statement.word_list:
                if word in i_l_n:
                    i_l_2.append(word)

        if len(i_l_1) > len(i_l_2):
            for w in i_l_1:
                if w in i_l_2:
                    result.append(w)
        else:
            for w in i_l_2:
                if w in i_l_1:
                    result.append(w)

        return result

    def get_tcc(self):
        common_method_pair_num = 0
        total_method_pair_num = 0

        num_methods = len(self.sr_class.method_list)
        for i in range(0, num_methods):
            for j in range(i, num_methods):
                if i == j:
                    continue
                comman_attr = self.get_intersection_il(method1=self.sr_class.method_list[i], method2=self.sr_class.method_list[j], i_l=self.sr_class.field_list)
                if len(comman_attr) > 0:
                    common_method_pair_num += 1
                total_method_pair_num += 1
        if total_method_pair_num == 0:
            return 0
        return round(common_method_pair_num / total_method_pair_num, 2)

    def get_lcom(self):
        result = 0

        i_l = []
        P = 0
        Q = 0
        num_methods = len(self.sr_class.method_list)
        field_list = self.sr_class.field_list
        for f in field_list:
            # if "static" not in f.modifiers:
                i_l.append(f)

        for i in range(0, num_methods):
            for j in range(i, num_methods):
                if i == j:
                    continue
                intersection_il = self.get_intersection_il(method1=self.sr_class.method_list[i], method2=self.sr_class.method_list[j], i_l=i_l)
                if len(intersection_il) > 0:
                    Q += 1
                else:
                    P += 1
        if P > Q:
            result = P - Q
        else:
            result = 0
        return result

    def get_dcc(self, class_list):
        dcc_count = 0
        class_name_list = [o.class_name for o in class_list]

        for field in self.sr_class.field_list:
            if field.field_type in class_name_list:
                dcc_count += 1

        for method in self.sr_class.method_list:
            for param in method.param_list:
                if param.type in class_name_list:
                    dcc_count += 1
        return dcc_count

    def get_cam(self):
        all_param_l = []
        intersection_param = []

        num_methods = len(self.sr_class.method_list)
        for method in self.sr_class.method_list:
            for param in method.param_list:
                all_param_l.append(param.type)
        for i in range(0, num_methods):
            for j in range(i, num_methods):
                if i == j:
                    continue
                p_l_i = [p.type for p in self.sr_class.method_list[i].param_list]
                p_l_j = [p.type for p in self.sr_class.method_list[j].param_list]

                i_l = []
                for p in p_l_i:
                    if p in p_l_j:
                        i_l.append(p)
                if len(i_l) > 0:
                    intersection_param.extend(i_l)
        if len(all_param_l) == 0:
            return 0
        return round(len(intersection_param) / len(all_param_l), 2)

    def get_dit(self, class_list):
        count = self.get_parent_count(self.sr_class, 0, class_list)
        return count

    def get_parent_count(self, sr_class, current_count, class_list):
        count = current_count
        if len(sr_class.extends) > 0:
            parent_class_name = sr_class.extends[1]
            parent_cls = None
            count += 1
            for cls in class_list:
                if cls.class_name == parent_class_name:
                    parent_cls = cls
            if parent_cls is None:
                return count
            else:
                return self.get_parent_count(parent_cls, count, class_list)
        else:
            return count

    def get_noam(self):
        noam = 0
        f_n_l = [o.field_name.lower() for o in self.sr_class.field_list]


        for m in self.sr_class.method_list:
            if m.method_name.startswith("get") or m.method_name.startswith("set"):
                m_l = m.method_name.split("get")
                if len(m_l) > 1:
                    if m_l[1].lower() in f_n_l:
                        noam += 1
                else:
                    m_l = m.method_name.split("set")
                    if m_l[1].lower() in f_n_l:
                        noam += 1
        return noam

    def get_method_noav(self, sr_method):
        fstl = []
        total_var_l = []
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for obj in fstl:
            for var in obj["var_name_l"]:
                if var not in total_var_l:
                    total_var_l.append(var)
        noav = len(total_var_l) + len(sr_method.param_list)
        return noav

    def get_method_block_depth(self, sr_method):
        self.calculate_depth(sr_method=sr_method)
        dp = 0
        for st in sr_method.get_all_statement():
            if st.block_depth > dp:
                dp = st.block_depth
        return dp

    def get_method_fuc(self, sr_method):
        result = 0
        for statement in sr_method.statement_list:
            s_re = self.get_statement_fuc(statement)
            result += s_re
        return result

    def get_method_lmuc(self, sr_method):
        result = 0
        for statement in sr_method.statement_list:
            s_re = self.get_statement_lmuc(statement)
            result += s_re
        return result
