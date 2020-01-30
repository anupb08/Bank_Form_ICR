import csv

rows = {}
def read_csv(filename):
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)

        for row in csvreader:
            rows[row[1]] = row


def proccess_field_information(info_file):
    fields_info = []
    with open(info_file) as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)

        for row in csvreader:
            fields_info.append(row)

    return fields_info



def parameters_of_matched_field(field, pos):
    param = None
    for row in rows:
        string = row.split('AND')
        if len(string) > 1:
            if string[0] in field and string[1] in field:
                param = rows[row]
                position = positions_range[param[2]]
                if pos[0] > position[0][0] and pos[0] < position[0][1] and pos[1] > position[1][0] and pos[1] < position[1][1]:
                    break
                else:
                    param = None
        elif string[0] in field:
            param = rows[row]
            position = positions_range[param[2]]
            print(string[0], field, pos, position)
            if pos[0] > position[0][0] and pos[0] < position[0][1] and pos[1] > position[1][0] and pos[1] < position[1][1]:
                break
            else:
                param = None
    return param



def is_print_on_page_enable():
    if configuration['print_on_page'] == '1' or configuration['print_on_page'] == 'True':
        return True

    return False

def get_page_nos():
    return eval(configuration['page_nos'])

def is_box_mark():
    if configuration['box_mark'] == '1' or configuration['box_mark'] == 'True':
        return True
    return False

def deskew_enabled():
    if configuration['enable_deskew'] == '1' or configuration['enable_deskew'] == 'True':
        return True
    return False

positions_range = {}
def get_positions():
    if 'positions' in configuration:
        positions = eval(configuration['positions'])
        for pos in positions:
            if pos in configuration:
                positions_range[pos] = eval(configuration[pos])

    return positions_range

def get_tesseract_command():
    return eval(configuration['command'])

configuration = {}
def read_application_config(configfile):
    with open(configfile) as config:
        #csvreader = csv.reader(config, delimiter='\t')
        for row in config:
            row = row.split('>')
            configuration[row[0].strip()] = row[1].strip()
    get_positions()
    return

def get_line_tolerance():
    line_tolerance = (8,3,30)
    if 'line_tolerance' in configuration:
        line_tolerance = eval(configuration['line_tolerance'])

    return line_tolerance

def get_segment_tolerance():
    seg_tolerance = (5, 2, 10)
    if 'segment_tolerance' in configuration:
        seg_tolerance = eval(configuration['segment_tolerance'])

    return seg_tolerance

def get_gap_for_page():
    gaps = {}
    gaps[1] = []
    if 'position_page_1' in configuration:
        pos = eval(configuration['position_page_1'])
        gaps[1].append(pos)
    if 'text_page_1' in configuration:
        text = eval(configuration['text_page_1'])
        gaps[1].append(text)


    return gaps




