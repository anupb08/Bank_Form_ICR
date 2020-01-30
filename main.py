from bank_form_process import read_configuration, pdf2image,process_bank_form
import time
import os
import sys

def main():
    t0 = time.time()
    print(t0)
    read_configuration()
    pdffile = sys.argv[1]
    if os.path.isfile(pdffile) is True:
        output = pdf2image(pdffile, 'bank.png', 'Result')
        print('output pdf', output)
    else:
        print("pdf file does not exist")
    #imgfile= 'Result/bank.png'
    #imgfile= 'temp2.png'
    #process_bank_form(imgfile)

    t8 = time.time()
    print('Total time time taken by draw_lines: %s' % (t8 -t0))


if __name__ == '__main__':
    main()
