import math 

def get_maxes(d: dict) -> str:
    current_max_val = float('-inf')
    current_max_key = ''

    k = d.keys()

    for i in k:
        if d[i] > current_max_val:
            current_max_val = d[i]
            current_max_key = i 
        elif d[i] == current_max_val:
            if (current_max_key == 'yes') or (i == 'yes'):
                current_max_key = 'yes'
    return current_max_key

    

def density(entry: float, mean: float, sd: float) -> float:  
    '''
    Returns the probability of the thing 
    '''

    power = -0.5 * pow(((entry - mean)/sd), 2)

    coef = (1) / (sd * math.sqrt(2 * math.pi))
    # print(f"Power: {power}, coef: {coef}, res: {coef * math.exp(power) }")
    # print(coef * math.exp(power) )
    return coef * math.exp(power) 

def partition(training, outcome) -> list:
    '''
    returns a partitioned list for a certain outcome
    '''
    new_list = [] 
    for i in training: 
        if i[-1] == outcome:
            new_list.append(i)
    return new_list 

def get_mean(training: list, set_length: int) -> list:
    mu_term = [0]*set_length
    for i in range(len(training)):
        # print(training[i])
        # Iterates through all training data
        for k in range(set_length):
            # Iterates through the mu_list
            mu_term[k] += float(training[i][k])
    for _ in range(set_length): 
        mu_term[_] = mu_term[_]/len(training)
    return mu_term 
    
def get_sd(training: list, set_length: int, mu_term: list) -> list:
    sd_term = [0]*set_length
    for i in range(len(training)):
        # iterates through the training data 

        for j in range(set_length):
            # Iterates through the sd_term

            sd_term[j] += ((float(training[i][j]) - mu_term[j]) * (float(training[i][j]) - mu_term[j]))
    
    for _ in range(set_length): 
        var = (sd_term[_] / (len(training) - 1))
        if var == 0:
            sd_term[_] = 1e-9
        else:
            sd_term[_] = math.sqrt(var)
        
    return sd_term 

def calculate(training: list, testing: list) -> float:
    '''
    Calculates the naive bayes for the entry 
    '''
    set_length = len(training[0]) - 1
    
    # Mu_term is a list of all the means of the columns    
    mu_term = get_mean(training, set_length)

    # sd_term is a list of all the sd's of the columns
    sd_term = get_sd(training, set_length, mu_term)
    # print(training)
    # print()
    # print(mu_term)

    # print()

    # print(sd_term)

    to_return = 1
    for i in range(set_length): 
        output = density(float(testing[i]), mu_term[i], sd_term[i])
        # print(f"testing {testing[i]}\nmu: {mu_term[i]}\nsd: {sd_term[i]}\noutput: {output}")

        to_return *= output
        # print(f", toReutnr: {to_return}")

    return to_return 

def classify_nb(training_filename, testing_filename, toggle = True):
    # open files 
    if toggle:
        training_file = open(training_filename, 'r')
        testing_file = open(testing_filename, 'r')

        # read in training data and store as lists. 

        training_contents = training_file.readlines()
        testing_contents = testing_file.readlines() 

        training_file_contents = [x.strip().split(",") for x in training_contents]
        testing_file_contents = [y.strip().split(",") for y in testing_contents]
    else:
        training_file_contents = training_filename 
        testing_file_contents = testing_filename
    # Last column is class attribute 
    class_attribute_index = len(training_file_contents[0])-1

    # Class set is the set of all available output classes
    class_set = set([x[class_attribute_index] for x in training_file_contents])

    # d = dictionary that holds each class and probability associated with each 
    # d = {class : P(class | E)}
    # E.g. {Yes: 0.4; No: 0.3}

    to_return = [] 
    d = dict()
    outcome_dict = dict() 
    

    # Makes a dictionary of each output class and a list of all the
    # Data that adheres to it. 
    for _ in class_set: 
        d[_] = 1
        outcome_dict[_] = partition(training_file_contents, _)
    
    # This stores P(Class)
    dict_of_probs = dict() 
    for _ in class_set: 
        dict_of_probs[_] = len(outcome_dict[_]) / len(training_file_contents)


    for iTesting in testing_file_contents:
        # Each attribute will have to be evaluated.
        for outcome in class_set: 
            # print(f"OUTCOME: {outcome}")
            result = calculate(outcome_dict[outcome], iTesting)
            # Here, we look through the training data to evaluate. 
            # Need to calculate :
            # P(E | class) = P(E1 | class) * P(E2 | class) * ...
            # print(f"RESULT: {result}\n--------\n\n\n")
            d[outcome] = result
        
        class_outputs = {}

        
        for outcome in d.keys():
            # print(f'Outcome: {outcome}, d[outcome]: {d[outcome]}, dict_of_probs[otucome]: {dict_of_probs[outcome]}')
            # print(f"Output here: {d[outcome] * dict_of_probs[outcome]}")
            class_outputs[outcome] = d[outcome] * dict_of_probs[outcome]
        
        
        # print(f'Class_outputs: {class_outputs}')
        to_return.append(get_maxes(class_outputs))
        # print("_____________")    
    
    return to_return


if __name__ == "__main__":
    print(classify_nb("training_1.csv", "testing_1.csv"))