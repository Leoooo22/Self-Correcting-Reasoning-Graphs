from datasets import load_dataset
from collections import Counter
import ast

mbpp = load_dataset("google-research-datasets/mbpp", "sanitized")
#print(mbpp)

t = mbpp["test"][0]["test_list"][0]
print(type(t))
print(t)
print(ast.dump(ast.parse(t).body[0], indent=2))


#test = mbpp["test"]
"""lengths = [len(problem["test_list"]) for problem in test]
print(set(lengths))
print(Counter(lengths))

imports = [problem for problem in test if problem["test_imports"] != []]
print(len(imports))


for problem in test:
    for t in problem["test_list"]:
        if "==" not in t:
            print(t)
"""