class FPTreeNode():

    def __init__(self, name):
        self.name = name
        self.count = 0
        self.parent = None
        self.children = []
        self.next_homonym = None
        self.tail = None

    def add_child(self, child):
        self.children.append(child)

    def find_child(self, name):
        for child in self.children:
            if child.name == name:
                return child
        return None


class FPGrowth():

    def __init__(self, dataset, min_sup=0.0, min_conf=0.0):
        self.dataset = dataset
        self.min_sup = min_sup
        self.min_conf = min_conf
        self.freq_L1 = {}  # 1-频繁项
        self.freq_itemsets = []  # 存储每个频繁项及其对应的计数
        self.strong_association_rules = []  # 存储强关联规则

    def __get_frequency(self, trans_records):
        rect = {}
        for line in trans_records:
            for item in line:
                rect[item] = rect.get(item, 0)+1
        return rect

    def build_fptree(self):
        if self.dataset is None:
            return
        # 依据销售数量创建item序列
        self.freq_L1 = self.__get_frequency(self.dataset)
        tmp_list = []
        tmp_list.extend(self.freq_L1.keys())
        tmp_list.sort(key=lambda x: self.freq_L1[x])
        tmp_dict = {}
        i = 1
        for item in tmp_list:
            tmp_dict[item] = i
            i += 1
        # 依据item序列编号排序
        for trans_record in self.dataset:
            trans_record.sort(key=lambda x: tmp_dict[x], reverse=True)
        self.__fpgrowth(self.dataset, [])
        # 对结果排序
        self.freq_itemsets.sort(key=lambda x: len(x[0]), reverse=False)

    def __fpgrowth(self, cpb, post_model):
        freq_dict = self.__get_frequency(cpb)
        headers = {}
        data_num = len(self.dataset)
        for key in freq_dict:
            # 每一次递归时都有可能出现一部分模式的频数低于阈值
            if freq_dict.get(key) / data_num >= self.min_sup:
                node = FPTreeNode(key)
                node.count = freq_dict[key]
                headers[key] = node
        tree_root = self.__build_subtree(cpb, headers)
        if len(tree_root.children) == 0:
            return
        for header in headers.values():
            rule = []
            rule.append(header.name)
            rule.extend(post_model)
            # 表头项+后缀模式  构成一条频繁模式（频繁模式内部也是按照F1排序的），频繁度为表头项的计数
            temp = (rule, header.count)
            self.freq_itemsets.append(temp)
            # 新的后缀模式：表头项+上一次的后缀模式（注意保持顺序，始终按F1的顺序排列）
            new_post_pattern = []
            new_post_pattern.append(header.name)
            new_post_pattern.extend(post_model)
            # 新的条件模式基
            new_CPB = []
            next_node = header
            while True:
                next_node = next_node.next_homonym
                if next_node is None:
                    break
                count = next_node.count
                # 获得从虚根节点（不包括虚根节点）到当前节点（不包括当前节点）的路径，即一条条件模式基。注意保持顺序：父节点在前，子节点在后，即始终保持频率高的在前
                path = []
                parent = next_node
                while True:
                    parent = parent.parent
                    # 虚根节点的name为null
                    if parent.name is None:
                        break
                    path.append(parent.name)
                path.reverse()
                # 事务要重复添加counter次
                while count > 0:
                    count -= 1
                    new_CPB.append(path)
            self.__fpgrowth(new_CPB, new_post_pattern)

    def __build_subtree(self, trans_records, headers):
        # 虚根节点
        root = FPTreeNode(None)
        for trans_record in trans_records:
            record = []
            record.extend(trans_record)
            subtree_root = root
            tmpRoot = None
            if len(root.children) != 0:
                # 延已有的分支，令各节点计数加1
                while len(record) != 0:
                    tmpRoot = subtree_root.find_child(record[0])
                    if tmpRoot is None:
                        break
                    tmpRoot.count += 1
                    subtree_root = tmpRoot
                    record.pop(0)
            # 长出新的节点
            self.__add_nodes(subtree_root, record, headers)
        return root

    def __add_nodes(self, ancestor, record, headers):
        while len(record) > 0:
            item = record.pop(0)
            if item in headers:
                leafnode = FPTreeNode(item)
                leafnode.count = 1
                leafnode.parent = ancestor
                ancestor.add_child(leafnode)
                header = headers[item]
                tail = header.tail
                if tail is None:
                    header.next_homonym = leafnode
                else:
                    tail.next_homonym = leafnode
                header.tail = leafnode
                self.__add_nodes(leafnode, record, headers)

