import sentencepiece as spm
import glob

book_list = glob.glob("/Users/rachel/PycharmProjects/Transformer/books/*.txt")

def load_text_file(filename):
    lines = []
    with open(filename, 'r') as fin:
        for line in fin:
            if line != '':
                lines.append(line.strip())
    lines = ' '.join(lines)
    lines = lines.split('. ')
    lines = [a for a in lines if len(a) > 0]
    return lines


def load_books(book_list):
    lines = []
    for book in book_list:
        lines += load_text_file(book)
    return lines


lines = load_books(book_list)
with open('books.txt', 'w') as fout:
    for line in lines:
        print(line, file=fout)

spm.SentencePieceTrainer.train(input='books.txt',
                               model_prefix='m',
                               vocab_size=1024,
                               user_defined_symbols=["<sos>",
                                                    "<eos>",
                                                    "<pad>"])