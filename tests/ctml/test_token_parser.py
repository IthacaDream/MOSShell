from ghoshell_moss.ctml.token_parser import CTMLParser
from collections import deque


def test_token_parser_baseline():
    q = deque()
    parser = CTMLParser(callback=q.append)
    content = "<foo><bar/>h</foo>"
    with parser:
        for c in content:
            parser.feed(c)
        parser.end()
    assert not parser.is_running()
    assert parser.is_done()
    assert len(parser.parsed()) == 5
    assert parser.buffer() == content
    assert len(q) == 5

    tokens = parser.parsed()
    i = 0
    for token in tokens:
        i += 1
        assert token.idx == i


def test_token_parser_default_parse():
    content = "<foo><bar/>hello</foo>"

    # test case 2
    tokens = list(CTMLParser.parse(content))
    assert len(tokens) == 5

    i = 0
    for token in tokens:
        i += 1
        assert token.idx == i
