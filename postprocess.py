


addrs = {
        0: 'House/Plot No',
        1: 'Floor',
        2: 'Apartment/Building Name',
        3: 'Line 2',
        4: 'Line 3',
        5: 'Street/Road',
        6: 'Landmark',
        7: 'City/Town/Village',
        8: 'Sector/Locality',
        9: 'District',
        10:'State/U.T. Code',
        11: 'ISo 3316 Country Code',
        12: 'PIN/Post Code',
        13: 'State',
        14:'STD Code',
        15:'Tel',
        16:'Mobile'
        }

def process_personal_details():
    return []

def process_residential_address(string):
    address = {}
    string = string.replace(';', ' ')
    texts_line = string.split('\n')
    for texts in texts_line:
        print("postprocessing-", texts)
        for text in texts.split(',,,'):
            fields = text.split('"')
            if len(fields) < 2:
                continue
            if 'plot no' in  text.lower():
                address[addrs[0]] = text.split('"')[1]
            elif 'floor' in text.lower():
                address[addrs[1]] = text.split('"')[1]
            elif 'building' in text.lower() or 'apartment' in text.lower():
                address[addrs[2]] = text.split('"')[1]
            elif 'line' in text.lower():
                address[text.split('"')[0]] = text.split('"')[1]
            elif 'street' in text.lower() or 'road' in text.lower():
                address[addrs[5]] = text.split('"')[1]
            elif 'landmark' in text.lower():
                address[addrs[6]] = text.split('"')[1]
            elif 'city' in text.lower() or 'town' in text.lower():
                address[addrs[7]] = text.split('"')[1]
            elif 'sector' in text.lower() or 'locality' in text.lower():
                address[addrs[8]] = text.split('"')[1]
            elif 'district' in text.lower():
                address[addrs[9]] = text.split('"')[1]
            elif 'U.T. code' in text.lower():
                address[addrs[10]] = text.split('"')[1]
            elif 'coutry code' in text.lower():
                address[addrs[11]] = text.split('"')[1]
            elif 'post code' in text.lower() or 'pin' in text.lower():
                address[addrs[12]] = text.split('"')[1]
            elif 'state*' in text.lower():
                address[addrs[13]] = text.split('"')[1]
            elif 'std code' in text.lower():
                address[addrs[14]] = text.split('"')[1]
            elif 'tel' in text.lower():
                address[addrs[15]] = text.split('"')[1]
            elif 'mobile' in text.lower():
                address[addrs[16]] = text.split('"')[1]

    return address


def process_company_address(string):
    address = {}
    texts_line = string.split('\n')
    address['Company Address'] = texts_line[0].replace('"', '').replace(',,,', '')
    for texts in texts_line[1:]:
        texts = texts.replace(';' , ' ')
        for text in texts.split(',,,'):
            fields = text.split('"')
            if len(fields) < 2:
                continue
            if 'landmark' in text.lower():
                address[addrs[6]] = text.split('"')[1]
            elif 'city' in text.lower() or 'town' in text.lower():
                address[addrs[7]] = text.split('"')[1]
            elif 'sector' in text.lower() or 'locality' in text.lower():
                address[addrs[8]] = text.split('"')[1]
            elif 'district' in text.lower():
                address[addrs[9]] = text.split('"')[1]
            elif 'U.T. code' in text.lower():
                address[addrs[10]] = text.split('"')[1]
            elif 'coutry code' in text.lower():
                address[addrs[11]] = text.split('"')[1]
            elif 'post code' in text.lower() or 'pin' in text.lower():
                address[addrs[12]] = text.split('"')[1]
            elif 'state*' in text.lower():
                address[addrs[13]] = text.split('"')[1]
            elif 'std code' in text.lower():
                address[addrs[14]] = text.split('"')[1]
            elif 'tel' in text.lower():
                address[addrs[15]] = text.split('"')[1]
            elif 'mobile' in text.lower():
                address[addrs[16]] = text.split('"')[1]
    #if len(texts_line) >= 6:
    #    pan = texts_line[5].split(';')
    #    address['PAN No. of Applicant'] = pan[0]
    #    address['PAN No. of Co Applicant'] = pan[1]

    return address
    

def process_employee_id(string):
    employee = {}
    string = string.replace(';', ' ')
    texts = string.split(',,,')
    print('process_employee_id :', string, 'Text', texts[0])
    employee['Employee ID'] = texts[0].split('"')[1]
    if len(texts) < 3:
        print("error in Employee field", texts)
        return employee
    employee['No of Years in Current job'] = texts[1].split('"')[1]
    employee['Total Experience(No of years)'] = texts[2].split('"')[1]

    return employee

def process_designation(string):
    designation = {}
    print('process_designation :', string)
    string = string.replace(';', ' ')
    texts = string.split(',,,')
    designation['Designation'] = texts[0].split('"')[1]
    if len(texts) < 2:
        print("error on designation field", string)
        return designation
    designation['Department'] = texts[1].split('"')[1]

    return designation

def process_name_fields(string):
    string = string.replace(';', ' ')
    names = {}
    texts = string.split('\n')
    names['Name'] = texts[0]
    if len(texts) < 5:
        print("error during process Name fields", string)
        return names
    names['Maiden Name'] = texts[1]
    names['Father Name'] = texts[2]
    names['Mother Name'] = texts[3]
    names['Spouse Name'] = texts[4]

    return names

def process_others(string):
    others = {}
    string = string.replace(';', ' ')
    texts = string.split(',,,')
    others['Others'] = texts[0].replace('"', '')
    if len(texts) < 2:
        print("error diuring process Others", string)
        return others
    others['Date of Birth'] = texts[1].split('"')[1]

    return others


def process_spouse(string):
    spouse = {}
    texts = string.split(';')
    if len(texts) < 2:
        print("error during process spouse information", string)
        return spouse
    spouse['Date of Birth of Spouse'] = texts[0]
    spouse['PAN of Spouse'] = texts[1]

    return spouse

def process_expiry_date(string):
    expiry = {}
    string = string.replace(';', ' ')
    texts = string.split('\n')
    if len(texts) < 2:
        print("error during process spouse information", string)
        return expiry
    expiry['Passport Expiry Date'] = texts[0]
    expiry['Driving Licence Expiry Date'] = texts[1]

    return expiry


def process_no_years(string):
    num_years = {}
    num_years['No of Years at above residency'] = string
    string = string.replace(';', ' ')
    texts = string.split(',,,')
    if len(texts) < 2:
        print("error during process spouse information", string)
        return num_years
    num_years['No of Years at above residency'] = texts[0].replace('"', '')
    num_years['If Rented, Mothly Rent'] = texts[1].replace('"', '')

    return num_years


def process_local_address(string, coords, side_string, side_pos, gap, keyText):
    address = {}
    address[keyText] = string
    val = ''
    texts = string.split('\n')
    if len(texts) != len(coords):
        return address
    print('process_local_address :',string, coords)
    for l_text, l_coord in zip(texts,coords):
        text = l_text.replace(';', '').replace('"', '').strip()
        if len(text) != len(l_coord):
            continue
        for txt,coord in  zip(list(text), l_coord): 
            for st, pos in zip(side_string,side_pos):
                for t, p in zip(st,pos):
                    if abs(p[0] +gap[0] - coord[0]) < 30 and abs(p[1] +gap[1]- coord[1])<20:
                        val += t + ' \'' + txt.replace('"', '') + '\', ' 
                        break
    if val.strip() != '':
        address[keyText] = val
    return address

def process_occupation_type(string,coords,side_string,side_pos,gap,keyText):
    address = {}
    address[keyText] = string
    val = ''
    texts = string.split('\n')
    if len(texts) != len(coords):
        return address
    print('process_occupation_type :',string, coords)
    for st, pos in zip(side_string,side_pos):
        if len(st) != len(pos):
            print('side_text and side_poition lengths length are not same of text :', string )
            continue
        for t, p in zip(st,pos):
            for l_text, l_coord in zip(texts,coords):
                text = l_text.replace(';', '').replace('"', '').strip()
                if len(text) != len(l_coord):
                    continue
                for txt,coord in  zip(list(text), l_coord): 
                    if abs(p[0] +gap[0] - coord[0]) < 30 and abs(p[1] +gap[1]- coord[1])<20:
                        val += t + ' \'' + txt.replace('"', '') + '\', ' 
                        break
    if val.strip() != '':
        address[keyText] = val
    return address

def process_related_person(value,coords,side_text,side_position,gap,keyText):
    return process_fields_option(value,coords,side_text,side_position,gap,keyText)

def process_fields_option(string,coords,side_string,side_pos,gap,keyText):
    options = {}
    options[keyText] = string
    print(string, coords)
    val = ''
    texts = string.split('\n')
    if len(texts) != len(coords):
        return options
    print('process_fields_option :',string, coords)
    for st, pos in zip(side_string,side_pos):
        if len(st) != len(pos):
            print('side_text and side_poition lengths length are not same of text :', string )
            continue
        for t, p in zip(st,pos):
            for l_text, l_coord in zip(texts,coords):
                text = l_text.replace(';', '').replace('"', '').strip()
                if len(text) != len(l_coord):
                    print('Number of fields and boxes are not same of text :', string )
                    continue
                for txt,coord in  zip(list(text), l_coord): 
                    if len(side_string) != len(side_pos):
                        continue
                    if abs(p[0] +gap[0] - coord[0]) < 30 and abs(p[1] +gap[1]- coord[1])<20:
                        val += t + ' \'' + txt.replace('"', '') + '\', ' 
                        break
    if val.strip() != '':
        options[keyText] = val
    return options

def process_variable_fields_option(string,coords,side_string,side_pos,gap,keyText):
    options = {}
    options[keyText] = string.replace(',,,', ',').replace('"', '\'')
    return options


