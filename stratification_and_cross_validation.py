import math 
from kmeans import classify_nn 
from nbayes import classify_nb

def print_data(data: list) -> None: 
    for i in data: 
        for j in range(len(i)):
            if j == len(i) - 1:
                print(i[j])
            else:
                print(f'{float(i[j])},', end="")

def stratification(name: str) -> list:
    '''
    stratifies a dataset and returns a list of folds
    '''
    f = open(name, 'r') 
    c = f.readlines() 
    contents = [x.strip().split(',') for x in c]
    # print_data(contents)

    yes_no_dict = {'yes':[], 'no': []}

    for d in contents: 
        yes_no_dict[d[-1]].append(d)    


    # print(f'Yes: {len(yes_no_dict['yes'])}, No: {len(yes_no_dict['no'])}')

    folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7:[], 8:[], 9:[], 10:[]}
    # print(folds)
    dict_keys = yes_no_dict.keys() 
    for k in dict_keys:
        i = 0 
        length = len(yes_no_dict[k])
        l = yes_no_dict[k]
        

        while i < length: 
            # print(f'i: {i}, l[i]: {l[i]}')
            # print(f'i: {i%10} at {folds}')
            folds[(i%10) + 1].append(l[i])
            # print(f'AFTER: {folds}')
            i += 1

    # print(folds)
    to_return = []
    for i in range(10):
        to_return.append(folds[i+1])
        # print(f"fold{i+1}")
        # print(folds[i+1])
        # print_data(folds[i+1])
        # print()

    return to_return

def extract(lst: list) -> list:
    '''
    Extracts the results column for te thing
    '''
    to_return = []
    for i in lst: 
        to_return.append(i[-1])

    return to_return 

def generate_matrix(expected: list, actual: list) -> list:
    '''
    generates confusion matrix
    '''
    
    yes_yes = 0 
    yes_no = 0 
    no_yes = 0 
    no_no = 0 

    for i in range(len(expected)):
        if (expected[i] == 'yes') and (actual[i] == 'yes'):
            yes_yes += 1
        elif (expected[i] == 'yes') and (actual[i] == 'no'):
            yes_no += 1
        elif (expected[i] == 'no') and (actual[i] == 'yes'):
            no_yes += 1
        elif (expected[i] == 'no') and (actual[i] == 'no'):
            no_no += 1
    conf = [[yes_yes, no_yes], [yes_no, no_no]]
    return conf 

def get_accuracy(expected: list, actual: list) -> float:
    '''
    Gets the accuracy between two lists
    '''
    total = len(actual)
    tally = 0
    for i in range(total):
        if expected[i] == actual[i]:
            tally += 1
    return tally/total

def make_test_data(lst: list) -> list:
    '''
    Makes the test list
    '''
    to_return = [] 
    for i in lst: 
        tmp = i[0:-1]
        to_return.append(tmp)
    return to_return

def precision(matrix: list) -> float: 
    '''
    [[tp,fn], [fp,tn]]

    '''
    return (matrix[0][0]) / (matrix[0][0] + matrix[1][0])

def recall(matrix:list) -> float: 
    return (matrix[0][0]) / (matrix[0][0] + matrix[0][1])

def f1(P, R):
    return (2*P*R) / (P+R)

def cross_validation(folds: list) -> dict:
    '''
    folds :: folds dataset. 
    outputs a dictionary of accuracies: 
    '''

    to_return = {'NB' : [], 'KM_1': [], 'KM_5': []}
    NB_p = []
    KM_1_p = []
    KM_5_p = []

    NB_r = []
    KM_1_r = []
    KM_5_r = []


    NB_f = [] 
    KM_1_f = []
    KM_5_f = []


    # run the classifiers
    for i in range(10):
        # i will be the test set. Others will be training set.
        temp_test = make_test_data(folds[i]) # remove last column 
        even_more_temporary_test = folds[i]
        temp_training = [] 
        for j in range(10):
            if j != i:
                temp_training += folds[j]

        # nb_dict = {'f1': 0, 'f2':0, 'f3':0, 'f4':0, 'f5':0, 'f6':0, 'f7':0, 'f8':0, 'f9':0, 'f10':0}
        # km_dict = {'f1': 0, 'f2':0, 'f3':0, 'f4':0, 'f5':0, 'f6':0, 'f7':0, 'f8':0, 'f9':0, 'f10':0}

        actual_nb_output = classify_nb(temp_training, temp_test, False)
        actual_km_output_1 = classify_nn(temp_training, temp_test, 1, False)
        actual_km_output_5 = classify_nn(temp_training, temp_test, 5, False)

        expected_output = extract(even_more_temporary_test)

        nb_accuracy = get_accuracy(expected_output, actual_nb_output)
        km_accuracy_1 = get_accuracy(expected_output, actual_km_output_1)
        km_accuracy_5 = get_accuracy(expected_output, actual_km_output_5)

        # print('='*20)
        # print(f'Iteration: {i} Confusion Matrix: NB')
        m1 = generate_matrix(expected_output, actual_nb_output)
        # for i in m1: 
        #     print(i)
        # print()
        p = precision(m1)
        r = recall(m1)
        f = f1(p, r)
        # print(f'Precision: {p}\nRecall: {r}\nF1: {f}')

        NB_p.append(p)
        NB_r.append(r)
        NB_f.append(f)

        # print(f'Iteration: {i} Confusion Matrix: KM')
        m2 = generate_matrix(expected_output, actual_km_output_1)
        

        # for i in m2: 
        #     print(i)
        # print()
        
        p = precision(m2)
        r = recall(m2)
        f = f1(p, r)
        # print(f'Precision: {p}\nRecall: {r}\nF1: {f}')

        KM_1_p.append(p)
        KM_1_r.append(r)
        KM_1_f.append(f)

        m3 = generate_matrix(expected_output, actual_km_output_5)
        p = precision(m3)
        r = recall(m3)
        f = f1(p, r)
        KM_5_p.append(p)
        KM_5_r.append(r)
        KM_5_f.append(f)
        # print(f'{actual_km_output}\n{expected_output}')
        # print(f'NB: {nb_accuracy}\nKM: {km_accuracy}')
        # print('='*20)

        to_return['NB'].append(nb_accuracy)
        to_return['KM_1'].append(km_accuracy_1)
        to_return['KM_5'].append(km_accuracy_5)

    x = [[NB_p, KM_1_p,KM_5_p], [NB_r, KM_1_r,KM_5_r], [NB_f, KM_1_f,KM_5_f]]
    return to_return, x

if __name__ == '__main__':
    pima_stratified = stratification('norm_diabetes.csv')

    room_stratified = stratification('norm_room.csv')

    print('Pima dataset evaluation: ')

    pima_res, x = cross_validation(pima_stratified)
    print(f'Raw results: {pima_res}')
    
    print(f'NB average: {sum(pima_res['NB']) / 10}\n1NN average: {sum(pima_res['KM_1']) / 10}\n5NN average: {sum(pima_res['KM_5']) / 10}\n')
    print(f'NB_p average: {sum(x[0][0]) / len(x[0][0])}\n1NN_p average: {sum(x[0][1]) / len(x[0][1])}\n5NN_p average: {sum(x[0][2]) / len(x[0][2])}\n')
    print(f'NB_r average: {sum(x[1][0]) / len(x[1][0])}\n1NN_r average: {sum(x[1][1]) / len(x[1][1])}\n5NN_r average: {sum(x[1][2]) / len(x[1][2])}')
    print(f'NB_f average: {sum(x[2][0]) / len(x[2][0])}\n1NN_f average: {sum(x[2][1]) / len(x[2][1])}\n5NN_f average: {sum(x[2][2]) / len(x[2][2])}')
    
    print("="*20)

    print('Room dataset evaluation: ')
    room_res, x = cross_validation(room_stratified)
    print(f'Raw results: {room_res}')
    print(f'NB average: {sum(room_res['NB']) / 10}\n1NN average: {sum(room_res['KM_1']) / 10}\n5NN average: {sum(room_res['KM_5']) / 10}\n')
    print(f'NB_p average: {sum(x[0][0]) / len(x[0][0])}\n1NN_p average: {sum(x[0][1]) / len(x[0][1])}\n5NN_p average: {sum(x[0][2]) / len(x[0][2])}\n')
    print(f'NB_r average: {sum(x[1][0]) / len(x[1][0])}\n1NN_r average: {sum(x[1][1]) / len(x[1][1])}\n5NN_p average: {sum(x[1][2]) / len(x[1][2])}\n')
    print(f'NB_f average: {sum(x[2][0]) / len(x[2][0])}\n1NN_f average: {sum(x[2][1]) / len(x[2][1])}\n5NN_p average: {sum(x[2][2]) / len(x[2][2])}\n')

    print("="*20)

