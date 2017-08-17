#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import io


def get_map(vocab_list_file=None, word_dict=None):
    ##########################################################################
    # get map <word_idx,word_name>
    with io.open(vocab_list_file, 'r', encoding='utf-8') as inputfile:
        for rawline in inputfile:
            # print rawline.strip('\n')
            line = rawline.strip('\n')
            cols = line.strip().split('\t')
            word_dict[int(cols[0])] = cols[1]


def get_topic_stats(word_topic_count_file=None, topic_stat=None, word_topic_table=None):
    ##########################################################################
    # Get topic stats
    with io.open(word_topic_count_file, 'r', encoding='utf-8') as inputfile:
        for rawline in inputfile:
            print rawline.strip('\n')
            line = rawline.strip('\n').split(' ')
            # print line
            for element in line:
                temp_array = element.split(':')
                if len(temp_array) == 2:
                    # print temp_array
                    topic_stat[int(temp_array[0])] = int(temp_array[1])

    for topIdx in topic_stat:
        # print str(topIdx) + " " + str(topic_stat[topIdx])
        # Init word_topic_table dict
        word_topic_table.update({topIdx: {}})


def normalize_probablities(word_topic_table_file=None, topic_stat=None, word_dict=None, word_topic_table=None,
                           readable_word_topic_file=None):
    ##########################################################################
    # normalize proba and create a human readable list for each topic with descendent proba
    # instead of the word-topic distrib table
    with io.open(word_topic_table_file, 'r', encoding='utf-8') as inputfile:
        for rawline in inputfile:
            # print rawline.strip('\n')
            # print "hello"
            line = rawline.strip('\n').split(' ')
            wordIdx = int(line[0])
            for element in line:
                temp_array = element.split(':')
                if len(temp_array) == 2:
                    # print temp_array
                    topicIdx = int(temp_array[0])
                    topicCount = topic_stat[topicIdx]
                    wtopicCount = int(temp_array[1])
                    proba = float(wtopicCount) / float(topicCount)
                    wname = word_dict[wordIdx]
                    word_topic_table[topicIdx].update({wname: proba})
    output_file = io.open(readable_word_topic_file, 'w', encoding='utf-8')
    probaThreshold = 0.006
    maxWordDistToPrint = 20
    for topIdx in word_topic_table:
        print "---------------------"
        output_file.write("---------------------" + '\n')
        topicTitle = "Topic " + str(topIdx)
        print topicTitle
        output_file.write(topicTitle + '\n')
        sum = 0
        counter = 0
        for w in sorted(word_topic_table[topIdx], key=word_topic_table[topIdx].get, reverse=True):
            proba = word_topic_table[topIdx][w]
            sum += proba
            if proba > probaThreshold or counter < maxWordDistToPrint:
                tempLine = str(counter + 1) + " " + w + "\t " + str(proba)
                print tempLine
                output_file.write(tempLine + '\n')
                counter += 1


def doc_topic_distribution(readable_doc_topic_file=None, doc_topic_table_file=None):
    ##########################################################################
    # doc-topic distribution --> Normalization only
    output_file = io.open(readable_doc_topic_file, 'w', encoding='utf-8')
    with io.open(doc_topic_table_file, 'r', encoding='utf-8') as inputfile:
        for rawline in inputfile:
            print rawline.strip('\n')
            # print "hello"
            line = rawline.strip('\n').split(' ')
            docIdx = int(line[0])
            output_line = str(docIdx) + ": \t "
            Norm = 0.0
            # get first the sum for normalizing proba
            for element in line:
                temp_array = element.split(':')
                if len(temp_array) == 2:
                    Norm += float(temp_array[1])
            for element in line:
                temp_array = element.split(':')
                if len(temp_array) == 2:
                    # print temp_array
                    topicIdx = int(temp_array[0])
                    doctopicCount = int(temp_array[1])
                    proba = 100.0 * float(doctopicCount) / float(Norm)

                    formated_topicIdx = "Topic %02d" % topicIdx
                    formated_proba = "%.2f %%" % proba
                    output_line += formated_topicIdx + " (" + formated_proba + ")\t "
            output_file.write(output_line + "\n")


def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("--dir", dest="data_directory", type=str, help="get the data directory",
                      default="/home/nw/quanox/data/rtl/output/")
    parser.add_option("--wtt", dest="word_topic_table_file", type=str, help="get the word topic table file name",
                      default="server_0_table_0.model")
    parser.add_option("--wtc", dest="word_topic_count_file", type=str, help="get the word topic count file name",
                      default="server_0_table_1.model")
    parser.add_option("--v", dest="vocabulary_list_file", type=str, help="get the vocabulary list file name",
                      default="QX_RTLDB_5000.word_id.dict")
    parser.add_option("--rwt", dest="readable_word_topic_file", type=str, help="get the readable word topic file name",
                      default="server_0_table_0.readable.model")
    parser.add_option("--dtt", dest="doc_topic_table_file", type=str, help="get the doc topic table file name",
                      default="doc_topic.0")
    parser.add_option("--rdt", dest="readable_doc_topic_file", type=str, help="get the readable doc topic file name",
                      default="doc_topic.0.readable.model")
    (options, args) = parser.parse_args()

    word_dict = {}
    topic_stat = {}
    # dic of dic:
    word_topic_table = {}

    get_map(vocab_list_file=options.vocabulary_list_file, word_dict=word_dict)
    get_topic_stats(word_topic_count_file=options.word_topic_count_file, topic_stat=topic_stat,
                    word_topic_table=word_topic_table)
    normalize_probablities(word_topic_table_file=options.word_topic_table_file, topic_stat=topic_stat,
                           word_dict=word_dict, word_topic_table=word_topic_table,
                           readable_word_topic_file=options.readable_word_topic_file)
    doc_topic_distribution(readable_doc_topic_file=options.readable_doc_topic_file,
                           doc_topic_table_file=options.doc_topic_table_file)


if __name__ == "__main__":
    main()
