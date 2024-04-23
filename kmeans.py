import typing 
import math 

def test_num(s: str) -> bool:
    try:
        float(s)
        return True 
    except ValueError:
        return False 

def dist(point1: list, point2: list) -> float:
    '''
    point1 :: list of numerics that is used to calculate distance 
    point2 :: list of numerics that is used to calculate distance 

    returns a float of the distance between the two points. 
    '''
    ans = 0 
    for i in range(len(point2)):
        t1 = point1[i]
        t2 = point2[i]

        if not(test_num(t1) or test_num(t2)):
            # i or j is non-numeric. This shuldn't be the case but whatever. 
            # We use the nominal attributes method
            if t1 == t2:
                ans += 0
            else:
                ans += 1 
            continue 
        # This is numeric attributes method
        ans += pow((float(t1) - float(t2)), 2)
    return math.sqrt(ans)

def most_common(lst: list) -> str: 
    d = {} 
    for i in lst:
        if i in set(d.keys()):
            d[i] += 1 
        else: 
            d[i] = 0 
    max_entry = max(d, key= d.get)
    return max_entry

def classify_nn(training_filename: str, testing_filename: str, k: int, toggle=True) -> list[str]:
    '''
    training_filename :: str of dataset used to train the classifier 
    testing_filename  :: str of dataset used to test the classifier 
    k :: int that is used to determine the number of nearest neighbours 

    The K-Nearest Neighbour algorithm should be implemented 
    for any K value and should use Euclidean distance 
    as the distance measure. In the case of ties, predict yes
    '''

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

    to_return = [] 
    

    for iTesting in testing_file_contents: 
        d = dict()
        # d = [distance(int) : class_lists(list[str])]
        # iTesting is a row of data in the testing file table 
        for iTraining in training_file_contents:
            # iTraining is a row of data in the training file table
            temp_distance = dist(iTraining, iTesting)

            class_attribute = iTraining[class_attribute_index]
            if temp_distance in d.keys():
                d[temp_distance].append(class_attribute)
            else:
                d[temp_distance] = [class_attribute]

        k_list = [] 
        length = 0 
        keyset = set(d.keys())
        while length < k: 
            smallest_key =  min(keyset)
            entry = d[smallest_key]
            k_list += entry
            keyset.remove(smallest_key)
            length += len(entry)
        
        k_list = k_list[:k]

        most_common_class = most_common(k_list)
        to_return.append(most_common_class)

    return to_return

if __name__ == "__main__":
    print(classify_nn("training_1.csv", "testing_1.csv", 2))
