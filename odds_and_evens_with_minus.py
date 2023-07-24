import numpy
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import copy

numpy.set_printoptions(threshold=numpy.inf) 

class Neural_Network(object):
    #   level                      层数
    #   levels_weights    每层权     （矩阵） 上一层神经元数 行， 这一层神经元数列
    #   levels_bias           每层偏量 （矩阵） 这一层神经元数 行，  1 列（竖着的numpy.matrix([[1],[2]])）
    def __init__(self, original_input, level, levels_weights, levels_bias, debug = False):
        self.level = level
        self.levels_weights = levels_weights
        self.levels_bias = levels_bias
        self.original_input = original_input
        
        if debug:
            print("   Neural network defined.")
            print("       Level\tNeurals")
            for i in range(-1, level):
                if i >= 0 :
                    print("       "+str(i)+"\t"+str(levels_bias[i].shape[0]))
                else:
                    print("       "+str(i)+"\t"+str(original_input.shape[0]),"\n")

    def calculate(self, levels_weights, levels_bias, new_input = None):
        if new_input is None:
            print("  Loaded empty input, so I used the input witch  you added it to me when you defining me.")
            new_input = self.original_input
            
        last_inputs = []
        last_input = numpy.array(new_input)
        last_inputs.append(last_input)
        for i in range(self.level):
            last_input = 1 / (1 + numpy.exp(numpy.dot(-levels_weights[i], last_input) + levels_bias[i]))
            #print(i,"\t",last_input.shape)
            last_inputs.append(last_input)
        
        #print(last_input)
        return last_input, last_inputs

    def train(self, training_set, val_set):
        l = 0.0000001    #  求导数用的步长，太小太大都会不准
        k = 0.1                           #  激励转化为移动的比例，太小速度慢，太大会炸
        
        verified_result = neural_network_test.verify(val_each_type_set, show_detail = False)
        print(verified_result)
        if verified_result['correct_rate'] == 1.0:
            #print(self.levels_weights)
            print("100 % correct rate has been reached. Press any key to improve the model.")
            input()
        
        tern_enthusiasm = 0
        for type_index in range(len(training_set)):                                   #  遍历训练集中每一类
            expected_result = numpy.zeros(self.levels_weights[-1].shape[0])  #  期望所得到的结果
            expected_result[type_index] = 1
            temp = expected_result * 2 - 1
            #temp = [n if n >= 0 else n / (temp.shape[0] - 1) for n in temp]
            controler = numpy.array([temp])
            controler = controler.reshape((controler.shape[1], controler.shape[0]))


            for each_sample in training_set[type_index]:                            #  遍历某一类中的所有图片
            
                #  先用验证集验证一下
                #print(neural_network_test.verify(val_each_type_set))
            
                sample_as_input = numpy.reshape(each_sample, [each_sample.shape[0] * each_sample.shape[1], 1])
                now_result = self.calculate(self.levels_weights, self.levels_bias, new_input = sample_as_input)
                
                #index_A = numpy.argwhere(expected_result == expected_result.max())[0]
                #index_B = numpy.argwhere(now_result[0] == now_result[0].max())[0]
                #enthusiasm = (numpy.abs(expected_result - now_result[0])).sum() * (1 if (index_A == index_B).all() else -1)
                standard_mistakes = ((expected_result - now_result[0]) ** 2 ).sum()
                ##  标准损失
                
                i_ls_w = []
                for weight_index in range(len(self.levels_weights)):           #  这块用来遍历 levels_weights
                    each_level_weights = self.levels_weights[weight_index]
                    i_each_l_w = numpy.zeros(each_level_weights.shape)      #  生成一个空的数组，用来储存偏导
                    
                    for i in range(each_level_weights.shape[0]):   
                        for j in range(each_level_weights.shape[1]):                            
                            test_weight = [numpy.array(e) for e in self.levels_weights]
                            test_weight[weight_index][i, j] = test_weight[weight_index][i, j] + l
                            
                            a = self.calculate(test_weight, self.levels_bias, sample_as_input)[0]
                            b = now_result[0]
                            #i_this_arg = ((a - b)/ l * controler).sum()
                            i_this_arg = self.levels_weights[-1].shape[0] ** 0.5 - (((a - b) - controler) ** 2).sum() ** 0.5
                            #print((((a - b) - controler) ** 2).sum() ** 0.5, i_this_arg)
                            #input()

                            i_each_l_w[i, j] = i_this_arg

                    i_ls_w.append(i_each_l_w)
                    
                i_ls_b = []
                for bias_index in range(len(self.levels_bias)):                #  这块用来遍历 levels_bias
                    each_level_bias = self.levels_bias[bias_index]
                    i_each_l_b = numpy.zeros(each_level_bias.shape)      #  生成一个空的数组，用来储存偏导
                    
                    for i in range(len(each_level_bias)):                          
                        test_bias = [numpy.array(e) for e in self.levels_bias]
                        test_bias[bias_index][i, 0] = test_bias[bias_index][i, 0] + l

                        a = numpy.array(self.calculate(test_weight, self.levels_bias, sample_as_input)[0])
                        b = numpy.array(now_result[0])
                        #i_this_arg = ((a - b) / l * controler).sum()
                        i_this_arg = self.levels_weights[-1].shape[0] ** 0.5 - (((a - b) - controler) ** 2).sum() ** 0.5

                        i_each_l_b[i, 0] = i_this_arg
                    i_ls_b.append(i_each_l_b)

                                #  计算矢量和绝对值
                i_vector_sum_abs = 0
                for i in range(len(i_ls_w)):
                    #print(i_ls_w[i].shape, i_ls_b[i].shape)
                    i_vector_sum_abs += (i_ls_w[i] ** 2).sum() +  (i_ls_b[i] ** 2).sum()
                i_vector_sum_abs = numpy.sqrt(i_vector_sum_abs)                
                
                
                #  得到单位向量  顺便 乘激励和系数
                if not i_vector_sum_abs == 0:
                    i_change_vector_w = []
                    i_change_vector_b = []
                    for i in range(len(i_ls_w)):
                        i_change_vector_w.append(i_ls_w[i] / i_vector_sum_abs * 1 * k)    #  这俩就是一会要改变的值
                        i_change_vector_b.append(i_ls_b[i] / i_vector_sum_abs * 1 * k)
                    
                    #  改变现在的参数
                    for i in range(len(i_ls_w)):
                        self.levels_weights[i] = self.levels_weights[i] + i_change_vector_w[i]
                        self.levels_bias[i] = self.levels_bias[i] + i_change_vector_b[i]
                    
                tern_enthusiasm += 0
        
        return tern_enthusiasm
     
    def verify(self, val_set, show_detail = False):
         result = {
              "correct_rate": 0,
              "delta": 0,
              "acc50": 0,
              "acc95": 0,
              "incorrect": [],
            }

         total_correct = 0
         total_tried = 0
         for i in range(len(val_set)):
            for e in val_set[i]:
                total_tried += 1
                get_calc = self.calculate(self.levels_weights, self.levels_bias, new_input = numpy.reshape(e, [e.shape[0] * e.shape[1], 1]))[0][:, 0]
                
                if show_detail:
                    print(i, get_calc[0], get_calc[1])

                max_index = numpy.argwhere(get_calc == get_calc.max())[0][0]
                min_index = numpy.argwhere(get_calc == get_calc.min())[0][0]
                result["delta"] += abs(get_calc[max_index] - get_calc[min_index])
                if max_index == i:
                    total_correct += 1
                    if get_calc[max_index] >= 0.5:
                        result["acc50"] += 1
                        if get_calc[max_index] >= 0.95:
                            result["acc95"] += 1
                #else:
                    #result['incorrect'].append(array_to_number(e))
                    
         result["correct_rate"] = total_correct / total_tried
         result["delta"] = result["delta"] / total_tried
         if not total_correct == 0:
             result["acc50"] = result["acc50"] / total_correct
             result["acc95"] = result["acc95"] / total_correct

         return result

def show_result_3d_diagram(result):
    #  得到结果中神经元最多的一层的神经元数
    neural_amount = [each.shape[0] for each in result]
    max_amount = max(neural_amount)
    
    
    #print(result)
    #  格式化 result
    formated_result = numpy.zeros([len(result), max_amount])
    for i in range(len(result)):
        formated_result[i,0:result[i].shape[0]] = result[i][:,0]

    fig =  plt.figure(figsize=(7, 7)) 
    ax  = fig.add_subplot(projection='3d')

    ens    = [ str(x) for x in range(max_amount)  ]
    colors = [(0.9*i%1,0.8*i%1,0.7*i%1) for i in range(len(result))]
    stas   = [ 'L'+str(x) for x in range(-1, len(result)-1)  ]
    stas[0], stas[-1] = "Input", "Result"
    colors[0], colors[len(result) - 1] = (0.8, 0.2, 0.1), (0.1, 0.8, 0.3)
    weight = formated_result


    for ith,ista in enumerate(stas):
        xs = np.arange(len(ens))
        ys = weight[ith]

        cs = [ colors[ith] ] * len(xs)
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax.bar(xs, ys, zs=ith, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('Neurals tag', fontsize=8)
    ax.set_ylabel('Neurals level', fontsize=10)
    ax.set_zlabel('Weight', fontsize=10)
    
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=8)
    ax.set_xticks(list(range(len(ens))))
    ax.set_xticklabels(ens)  
    ax.set_yticks(list(range(len(stas))))
    ax.set_yticklabels(stas)  
    ax.patch.set_facecolor('#ffffff')
    plt.show()

def generate_i_w_b(eachlevel_amount):
    levels_weights = []
    levels_bias = []
    for i in range(len(eachlevel_amount[1:])):  #  去除第一个，因为是输入值，不用管
        n = eachlevel_amount[i+1]
        father_n = eachlevel_amount[i]
        
        levels_weights.append(numpy.random.rand(n, father_n) * 2 - 1)
        levels_bias.append(numpy.random.rand(n,1) * 2 - 1)
        
    input_value = numpy.random.rand(eachlevel_amount[0],1)    #  输入值
    
    return input_value, levels_weights, levels_bias

def load_set(path):
    for root, dirs, files in os.walk(path):
        amount = len(files)
        loaded = numpy.zeros([amount, 16, 16])
        for i in range(len(files)):
            image_loaded = mpimg.imread(os.path.join(root, files[i]))
            loaded[i, :, :] = numpy.array(1 - image_loaded[:, :, 0:3].mean(axis = -1))
    return loaded


def number_to_array(number_list):
    result = []
    for num_raw in number_list:
        num = abs(num_raw)
        ten_stack = num // 10
        one_stack = num - 10 * ten_stack
        
        this_num = numpy.zeros((21, 1))
        this_num[one_stack, 0] = 1
        this_num[10 + ten_stack, 0] = 1
        
        this_num[20] = (0 if num_raw >= 0 else 1)
        result.append(this_num)

    return numpy.array(result)

def array_to_number(array):
                    switch = numpy.reshape(numpy.array([[0,1,2,3,4,5,6,7,8,9,0,10,20,30,40,50,60,70,80,90,0]]), (21,1))
                    num = numpy.sum(array * switch) *  (1 - array[20] * 2)
                    return num
if __name__ == "__main__":
    
    ##  2022.12.9 这是一个识别 0 - 99 以内的奇偶数的人工智能，只用神经网络，模拟人脑识别奇偶数

    input_value, levels_weights, levels_bias = generate_i_w_b([21, 22, 5])
    each_type_name = ["+Odds 正奇数", "+Evens 正偶数", "-Odds 负奇数", "-Evens 负偶数"]
    
    print("  Loading training set..")
    train_each_type_set = [[1,3, 5,7,9,11],[2,4,6,8,10], [-1,-3, -5,-7,-9,-11],[-2,-4,-6,-8,-10,]]

    val_each_type_set = [list(range(1,100,2)), list(range(2,100, 2)), list(range(-1,-100,-2)), list(range(-2,-100, -2))]
    
    #train_each_type_set = [list(range(1,100,2)), list(range(2,100, 2)), list(range(-1,-100,-2)), list(range(-2,-100, -2))]
    
    t = train_each_type_set
    v = val_each_type_set
    nta = number_to_array
    
    train_each_type_set = [nta(t[0]), nta(t[1]), nta(t[2]), nta(t[3])]
    val_each_type_set = [nta(v[0]), nta(v[1]), nta(v[2]), nta(v[3])]

    print("  Loaded!")
    print("\n      Training set:")
    for i, e in enumerate(train_each_type_set ):
        print("     ", e.shape,  each_type_name[i]) 
    print("\n      Val set:")
    for i, e in enumerate(val_each_type_set):
        print("     ", e.shape,  each_type_name[i]) 
    
    input()
    neural_network_test = Neural_Network(input_value, len(levels_weights), levels_weights, levels_bias, debug = True)
    print(neural_network_test.verify(val_each_type_set))
    
    times = 0
    while True:
        times += 1
        q = neural_network_test.train(train_each_type_set, val_each_type_set)
        print(times, end = "\t")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    while True:
        neural_network_test = Neural_Network(input_value, len(levels_weights), levels_weights, levels_bias, debug = True)
        numpy.set_printoptions(suppress=True, threshold = sys.maxsize, precision = 15)

        while True:
            
            result = neural_network_test.calculate(new_input = None)
            print(result[0])
            show_result_3d_diagram(result[1])

    
    
    
    
    
    
    
    
    
    
    
    
    
    """