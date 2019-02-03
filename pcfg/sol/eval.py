from grammar import *
from  cky import *

# ans should be True, False, False
def part_1():
    res = []
    print("Part 1:")
    with open('eval_gram_True.pcfg') as grammar_file:
        grammar = Pcfg(grammar_file)
        res.append(grammar.verify_grammar())
        #print(grammar.verify_grammar())

    with open('eval_gram_False.pcfg') as grammar_file:
        grammar = Pcfg(grammar_file)
        res.append(grammar.verify_grammar())
        #print(grammar.verify_grammar())

    with open('eval_gram_False_1.pcfg') as grammar_file:
        grammar = Pcfg(grammar_file)
        res.append(grammar.verify_grammar())
        #print(grammar.verify_grammar())
    return ["ANS:1 ", "Correct ans: " + str(res == [True, False, False]), "T F F" + str(res)]


# ans should be True, False
def part_2():
    print("Part 2:")
    with open('eval_gram_True.pcfg') as grammar_file:
        grammar = Pcfg(grammar_file)
    parser = CkyParser(grammar)
    test_sentence_1 = "the man saw the dog with the telescope".split(" ")
    test_sentence_2 = "the man the saw dog telescope with the".split(" ")
    # print(test_sentence_1, test_sentence_2)
    # print(parser.is_in_language(test_sentence_1))
    # print(parser.is_in_language(test_sentence_2))
    return ["ANS2: T F",parser.is_in_language(test_sentence_1), parser.is_in_language(test_sentence_2)]


# checks for table format
def part_3():
    print("Part 3:")
    with open('eval_gram_True.pcfg') as grammar_file:
        grammar = Pcfg(grammar_file)
    parser = CkyParser(grammar)
    test_sentence_1 = "the man saw the dog with the telescope".split(" ")
    table,probs = parser.parse_with_backpointers(test_sentence_1)
    return ["ANS3: ", "table format: " + str(check_table_format(table)), "prob format: " + str(check_probs_format(probs))]


# max probability parse tree is res, second best is res_false
def part_4():
    print("Part 4:")
    with open('eval_gram_True.pcfg') as grammar_file:
        grammar = Pcfg(grammar_file)
    parser = CkyParser(grammar)
    test_sentence_1 = "the man saw the dog with the telescope".split(" ")
    table,probs = parser.parse_with_backpointers(test_sentence_1)
    ans = get_tree(table, 0, len(test_sentence_1), grammar.startsymbol)
    res = ('TOP', ('NP', ('DT', 'the'), ('NN', 'man')), ('VP', ('VT', 'saw'), ('NP', ('NP', ('DT', 'the'), ('NN', 'dog')), ('PP', ('IN', 'with'), ('NP', ('DT', 'the'), ('NN', 'telescope'))))))
    res_false = ('TOP', ('NP', ('DT', 'the'), ('NN', 'man')), ('VP', ('VP', ('VT', 'saw'), ('NP', ('DT', 'the'), ('NN', 'dog'))), ('PP', ('IN', 'with'), ('NP', ('DT', 'the'), ('NN', 'telescope')))))

    return ["ANS4: ", res == ans, ans == res_false, ans ]

with open("res.txt", "w") as write_file:
    write_file.write(str(part_1()) + "\n")
    write_file.write(str(part_2()) + "\n")
    write_file.write(str(part_3()) + "\n")
    write_file.write(str(part_4()))







