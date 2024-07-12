# coding: utf8

from tqdm import tqdm
import numpy


def monitor(metrics, decimals):

    desc = ''

    for name, value in metrics:

        final_value = numpy.around(value, decimals=decimals)
        str_value = str(final_value)

        if '.' not in str_value:
            str_value += '.'
            str_value += ('0' * decimals)

        else:
            visible_dec = str_value.split('.')[1]
            padding = '0' * (decimals - len(visible_dec))
            str_value += padding

        desc += (name + ': ' + str_value + ' ')

    return desc
