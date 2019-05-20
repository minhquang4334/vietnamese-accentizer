import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
from io import open as open_unicode
from pyvi import ViTokenizer


dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = [u'.', u'!', u'?', u'...', u"'", u"`", u'"', dm_single_close_quote, dm_double_close_quote,
              u")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "corrector_dataset/train_set.txt"
all_val_urls = "corrector_dataset/val_set.txt"
all_test_urls = "corrector_dataset/test_set.txt"

finished_files_dir = "../finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


def chunk_file(set_name):
    in_file = 'finished_files/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print "Splitting %s data into chunks..." % set_name
        chunk_file(set_name)
    print "Saved chunked data in %s" % chunks_dir


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print "Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir)
    stories = os.listdir(stories_dir)
    # make IO list file
    print "Making list of files to tokenize..."
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print "Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir)
    subprocess.call(command)
    print "Stanford CoreNLP Tokenizer has finished."
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print "Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir)


def read_text_file(text_file):
    lines = []
    with open_unicode(text_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


def get_art_abs(lines):
    # Lowercase everything
    if (len(lines) != 2):
        raise Exception(
            "File have error format in line %s"%lines)

    noise_line = lines[0].encode('utf-8')
    norm_line = ViTokenizer.tokenize(lines[1])
    norm_line = norm_line.encode('utf-8')

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    noise_line = fix_missing_period(noise_line)
    norm_line = fix_missing_period(norm_line)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    norm_line = ' '.join(["%s %s %s" % (SENTENCE_START, norm_line, SENTENCE_END)])

    return noise_line, norm_line


def write_to_bin(url_file, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    print "Making bin file for texts in %s..." % url_file
    texts = read_text_file(url_file)

    with open(out_file, 'wb+') as writer:
        # Get the strings to write to .bin file
        for i in range(len(texts)):
            noise, norm = get_art_abs(texts[i].split('---'))
            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['noise'].bytes_list.value.extend([noise])
            tf_example.features.feature['norm'].bytes_list.value.extend([norm])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
        writer.close()

    print "Finished writing file %s\n" % out_file


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print "USAGE: python make_datafiles.py"
        sys.exit()

    # Create some new directories
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"))
    write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
    write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"))

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()
