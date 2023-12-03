import numpy as np

def zigzag(matrix: np.ndarray) -> np.ndarray:
    """
    computes the zigzag of a quantized block
    :param numpy.ndarray matrix: quantized matrix
    :returns: zigzag vectors in an array
    """
    # initializing the variables
    h = 0
    v = 0
    v_min = 0
    h_min = 0
    v_max = matrix.shape[0]
    h_max = matrix.shape[1]
    i = 0
    output = np.zeros((v_max * h_max))

    while (v < v_max) and (h < h_max):
        if ((h + v) % 2) == 0:  # going up
            if v == v_min:
                output[i] = matrix[v, h]  # first line
                if h == h_max:
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif (h == h_max - 1) and (v < v_max):  # last column
                output[i] = matrix[v, h]
                v = v + 1
                i = i + 1
            elif (v > v_min) and (h < h_max - 1):  # all other cases
                output[i] = matrix[v, h]
                v = v - 1
                h = h + 1
                i = i + 1
        else:  # going down
            if (v == v_max - 1) and (h <= h_max - 1):  # last line
                output[i] = matrix[v, h]
                h = h + 1
                i = i + 1
            elif h == h_min:  # first column
                output[i] = matrix[v, h]
                if v == v_max - 1:
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif (v < v_max - 1) and (h > h_min):  # all other cases
                output[i] = matrix[v, h]
                v = v + 1
                h = h - 1
                i = i + 1
        if (v == v_max - 1) and (h == h_max - 1):  # bottom right element
            output[i] = matrix[v, h]
            break
    return output

def inverse_zigzag(input, vmax, hmax):
        
        #print input.shape

        # initializing the variables
        #----------------------------------
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        output = np.zeros((vmax, hmax))

        i = 0
        #----------------------------------

        while ((v < vmax) and (h < hmax)): 
            #print ('v:',v,', h:',h,', i:',i)   	
            if ((h + v) % 2) == 0:                 # going up
                
                if (v == vmin):
                    #print(1)
                    
                    output[v, h] = input[i]        # if we got to the first line

                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1                        

                    i = i + 1

                elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                    #print(2)
                    output[v, h] = input[i] 
                    v = v + 1
                    i = i + 1

                elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                    #print(3)
                    output[v, h] = input[i] 
                    v = v - 1
                    h = h + 1
                    i = i + 1

            
            else:                                    # going down

                if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                    #print(4)
                    output[v, h] = input[i] 
                    h = h + 1
                    i = i + 1
            
                elif (h == hmin):                  # if we got to the first column
                    #print(5)
                    output[v, h] = input[i] 
                    if (v == vmax -1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1
                                    
                elif((v < vmax -1) and (h > hmin)):     # all other cases
                    output[v, h] = input[i] 
                    v = v + 1
                    h = h - 1
                    i = i + 1




            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                #print(7)        	
                output[v, h] = input[i] 
                break


        return output
    