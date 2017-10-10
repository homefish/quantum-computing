'''
This program implements quantum gates that are used in quantum computation.

References:
1. http://michaelnielsen.org/blog/quantum-computing-for-the-determined/
2. http://www.monoidal.net/papers/tutorialqpl-1.pdf
3. http://www.davyw.com/quantum/

Usage:
I recommend having a look at the test functions first, e.g. 
at the bottom of the program use

if __name__ == '__main__':
    test_controlled_not_2()
    
to run test_controlled_not_2() which illustrates an example
of the controlled_not gate.


You'll need Python3 to run this program, e.g. use 
python3.4 qc.py
to run the program in the terminal.
'''


import math


#-------------------------------------------------------------------


def to_bin(some_int, n):
    '''
    Returns the binary representation of some_int as string
    with zeros padded in front such that the total length of
    the string equals n.
    For example, if n = 8, then the binary representation of the
    integer 5 returned by this function is '00000101' 
    '''
    b = bin(some_int)[2:]
    s = (n - len(b)) * '0' + b
    return s

    
#-------------------------------------------------------------------    

    
def swap(state, x, y):
    '''
    This function swaps the x-th and y-th qubit and returns
    that new state. Note: 'state' is not mutated!
    
    Example: 3-qubit state initialized as 
    |psi> = 
    0.2         |000> + 
    (0.2 + 0.2i)|001> + 
    0.2         |010> + 
    (0.2-0.2i)  |011> + 
    (0.2+0.2i)  |100> + 
    0.2         |101> + 
    0.2         |110> + 
    sqrt(0.6)     |111>
    
    We store this state as a vector 
    [0.2, 0.2+0.2i, 0.2, 0.2-0.2i, 0.2+0.2i, 0.2, 0.2, sqrt(0.6)]
    
    If we swap the first and the last qubit, i.e. apply swap(psi, 0, 2), then the
    following happens to the base vectors of |psi>:
    
    We apply the swap to every basis vector:
    |000> becomes |000>     ---     in decimal: 0 becomes 0
    |001> becomes |100>     ---     in decimal: 1 becomes 4
    |010> becomes |010>     ---     in decimal: 2 becomes 2
    |011> becomes |110>     ---     in decimal: 3 becomes 6
    |100> becomes |001>     ---     in decimal: 4 becomes 1
    |101> becomes |101>     ---     in decimal: 5 becomes 5
    |110> becomes |011>     ---     in decimal: 6 becomes 3
    |111> becomes |111>     ---     in decimal: 7 becomes 7
    
    This means the following:
    If psi_vec is our state vector, then the new vector after the swap operation is:
    psi_vec_swapped = [ psi_vec[0], psi_vec[4], psi_vec[2], psi_vec[6], psi_vec[1], psi_vec[5], psi_vec[3], psi_vec[7] ]
    
    Strategy for implementation:
    1. Initialize a new vector of the same length as the vector for psi,
       i.e. a vector of length 2^{n}, where n is the number of qubits    
    2. Go through all integers from 0 to 2^{n-1}. For each integer i consider
       the binary representation b. 
    3. Perform the swap on b, e.g. swap(|110>, 0, 2) means
       110 becomes 011, i.e. in decimal representation 6 becomes 3.
       This corresponds to the 6th entry going to the 3rd entry of the new vector.
       Assign psi_vec_swapped[6] = psi_vec[3]
    '''

    # Step 1: 
    new_state = [0 for _ in range(len(state))]
    
    N = len(state)            # number of coefficients to represent the state
    n = int(math.log(N, 2))   # number of qubits

    # Step 2:
    for i in range(2**n):
        b = to_bin(i, n)
        L = list(b)
        L[x], L[y] = L[y], L[x]   # step 3
        swapped_b = "".join(L)
        new_i = int(swapped_b, 2)
        new_state[new_i] = state[i]
    return new_state    

#-----------------------------------------------------------------------------------------


def hadamard(state, i):
    '''
    This function applies the hadamard gate to the i-th qubit and returns
    that new state. Note: 'state' is not mutated!
    
    The hadamard gate does the following:
    H |0> = 1/sqrt(2) ( |0> + |1> )
    H |1> = 1/sqrt(2) ( |0> - |1> )

    Example 1:
    Consider a 3-qubit state. The basis vectors are
    |000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>
    and in general the a 3-qubit state is given by
    |psi> = a0 |000> + a1 |001> + a2 |010> + a3 |011>
          + a4 |100> + a5 |101> + a6 |110> + a7 |111>
    
    Let's apply the Hadamard gate to the first qubit at position 0:
    p_vec_new is initialized as [0, 0, 0, 0, 0, 0, 0, 0]

    a0 |000>        becomes         1/sqrt(2) a0 |000> + 1/sqrt(2) a0 |100>    --- in vector notation: p_vec_new[int("000",2)] += 1/sqrt(2) p_vec[int("000",2)] 
    a1 |001>        becomes         1/sqrt(2) a1 |001> + 1/sqrt(2) a1 |001>                        and p_vec_new[int("100",2)] += 1/sqrt(2) p_vec[int("000",2)]
    a2 |010>        becomes         1/sqrt(2) a2 |010> + 1/sqrt(2) a2 |110>        
    a3 |011>        becomes         1/sqrt(2) a3 |011> + 1/sqrt(2) a3 |111>
    a4 |100>        becomes         1/sqrt(2) a4 |000> - 1/sqrt(2) a4 |100>
    a5 |101>        becomes         1/sqrt(2) a5 |001> - 1/sqrt(2) a5 |101>
    a6 |110>        becomes         1/sqrt(2) a6 |010> - 1/sqrt(2) a6 |010>
    a7 |111>        becomes         1/sqrt(2) a7 |011> - 1/sqrt(2) a7 |111>
    
    Example 2:
    |psi > a |00> + b |10>
    H a|00> = 1/sqrt(2) a|00> + 1/sqrt(2) a |10>
    H b|10> = 1/sqrt(2) b|00> - 1/sqrt(2) b |10>
    So H |psi> = 1/sqrt(2) [(a+b) |00> + (a-b) |10>] 
    
    Strategy for implementation:
    for num=0 to 2^n - 1:
        b = binary representation of num
        b0 = binary_representation of num with 0 in i-th position
        b1 = b0 with 1 in i-th position
        
        if b has a zero in i-th position:
            psi_new_vec[int(b0)] += 1/sqrt(2) psi_vec[num]
            psi_new_vec[int(b1)] += 1/sqrt(2) psi_vec[num]
        else:
            psi_new_vec[int(b0)] += 1/sqrt(2) psi_vec[num]
            psi_new_vec[int(b1)] -= 1/sqrt(2) psi_vec[num] 
    '''
    # Step 1: 
    new_state = [0.0 for _ in range(len(state))]
    
    N = len(state)            # number of coefficients to represent the state
    n = int(math.log(N, 2))   # number of qubits

    # Step 2:
    for num in range(2**n):
        b = to_bin(num, n)
        L = list(b)
        
        L_b0 = L[:]
        L_b0[i] = '0'
        b0 = "".join(L_b0)
        
        L_b1 = L[:]
        L_b1[i] = '1'
        b1 = "".join(L_b1)
        
        if b[i] == '0':
            new_state[int(b0, 2)] += 1.0/math.sqrt(2) * state[num]
            new_state[int(b1, 2)] += 1.0/math.sqrt(2) * state[num]
        else:
            new_state[int(b0, 2)] += 1.0/math.sqrt(2) * state[num]
            new_state[int(b1, 2)] -= 1.0/math.sqrt(2) * state[num]
        # print(new_state)
            
    return new_state
                
              
#----------------------------------------------------------------------------------------


def not_gate(state, i):
    '''
    This function applies the not_gate to the i-th qubit and returns
    that new state. Note: 'state' is not mutated!
    
    The action of the not_gate is the following:
    not_gate(|0>) = |1>
    not_gate(|1>) = |0>
    
    Example 1:
    Consider a 3-qubit state. The basis vectors are
    |000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>
    and in general the a 3-qubit state is given by
    |psi> = a0 |000> + a1 |001> + a2 |010> + a3 |011>
          + a4 |100> + a5 |101> + a6 |110> + a7 |111> 
          
    Let's apply not_gate to the qubit in position 0.
    not_gate(|psi>) = not_gate(a0 |000>) + not_gate(a1 |001>) + not_gate(a2 |010>) + not_gate(a3 |011>)
                    + not_gate(a4 |100>) + not_gate(a5 |101>) + not_gate(a6 |110>) + not_gate(a7 |111>)
                      
    a0 |000>        becomes         a0 |100>
    a1 |001>        becomes         a1 |101>
    a2 |010>        becomes         a2 |110>
    a3 |011>        becomes         a3 |111>
    a4 |100>        becomes         a4 |000>
    a5 |101>        becomes         a5 |001>
    a6 |110>        becomes         a6 |010>
    a7 |111>        becomes         a7 |011>
    
    Strategy for implementation of not_gate(state, i)
    initialize new_state to [0, ..., 0], where new_state is a list of length 2^n
    for num=0 to 2^n - 1:
        b = binary representation of num
        if b[i] == '0': new_b = b with i-th position set to '1'
        else: new_b = b with i-th position set to '0'
        new_state[int(new_b)] += state[num]
    return new_state
    '''
    
    # Step 1: 
    new_state = [0.0 for _ in range(len(state))]
    
    N = len(state)            # number of coefficients to represent the state
    n = int(math.log(N, 2))   # number of qubits

    # Step 2:
    for num in range(2**n):
        b = to_bin(num, n)
        L = list(b)
        if b[i] == '0': L[i] = '1'
        else: L[i] = '0'
        new_b = "".join(L)
        new_state[int(new_b, 2)] += state[num]
        
    return new_state


#----------------------------------------------------------------------------------------

def controlled_not(state, x, y):
    '''
    This function applies the controlled_not gate to the x-th and y-th qubit
    of 'state' and returns that new state. 
    Note: 'state' is not mutated!
    
    The action of controlled_not is as follows:
    controlled_not( |...x...y...>)
    if x == 1:
        y is flipped, i.e. if y is 0 it becomes 1, and if y is 1 it becomes 0.
    else:
        nothing happens to the state.
        
    In a sense x acts as a switch. If the switch is on, i.e. x is 1, then
    y is flipped. Else, i.e. x is 0, nothing happens to y.
    See: https://en.wikipedia.org/wiki/Controlled_NOT_gate
    
    Example 1:
    Consider the 2-qubit state  a0 |00> + a1 |10>.
    Let's assume that the control qubit is on the left at position 0, 
    and the target qubit is on the right at position 1.
    
    Then the action of controlled_not is:
    controlled_not( state, 0, 1) = a0 controlled_not(|00>) + a1 controlled_not(|10>)
                                 = a0 |00> + a1 |11>
                                 
                                 
    Example 2:
    Consider a 3-qubit state. The basis vectors are
    |000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>
    and in general the a 3-qubit state is given by
    |state> = a0 |000> + a1 |001> + a2 |010> + a3 |011>
          + a4 |100> + a5 |101> + a6 |110> + a7 |111> 

    The action of controlled_not(state, 0, 2), where the the control qubit is 
    on the left at position 0 and the target qubit is on the right at position 2, is:
    a0 |000>        becomes         a0 |000>
    a1 |001>        becomes         a1 |001>
    a2 |010>        becomes         a2 |010>
    a3 |011>        becomes         a3 |011>
    a4 |100>        becomes         a4 |101>
    a5 |101>        becomes         a5 |100>
    a6 |110>        becomes         a6 |111>
    a7 |111>        becomes         a7 |110>
    
    
    Strategy for implementation of controlled_not(state, x, y):
    Initialize new_state to [0, ..., 0], where new_state is a list of length 2^n,
    where n is the number of qubits.
    for num=0 to 2^n - 1:
        b = binary representation of num
        if b[x] == '0': 
              new_state[num] += state[num]
        else:
            new_b = b with y-th position flipped 
            new_state[int(new_b)] += state[num]
    return new_state
    '''

    # Step 1: 
    new_state = [0.0 for _ in range(len(state))]
    
    N = len(state)            # number of coefficients to represent the state
    n = int(math.log(N, 2))   # number of qubits

    # Step 2:
    for num in range(2**n):
        b = to_bin(num, n)
        if b[x] == '0':
            new_state[num] += state[num] 
        else:
            L = list(b)
            if L[y] == '0': L[y] = '1'
            else: L[y] = '0'
            new_b = "".join(L)
            new_state[int(new_b, 2)] += state[num]
    return new_state


#----------------------------------------------------------------------------------------


def print_state_with_basis_vectors(state):
    N = len(state)
    n = int(math.log(N, 2))
    EPSILON = 0.001
    
    for idx, coeff in enumerate(state):
        #if coeff == 0.0:
        #    continue
        if (coeff.real) * (coeff.real) + (coeff.imag) * (coeff.imag) < EPSILON * EPSILON:
            continue
        b = to_bin(idx, n)
        z = coeff
        probability = z.real * z.real + z.imag * z.imag
        
        z_print = round(z.real, 3) + round(z.imag, 3) * 1j
        print(z_print , "\t|", b, "> \t probability: ", round(probability * 100,1), "%")


def new_state(num_qubits):
    '''
    returns a list of length 2^n of the form [1, 0, ..., 0] that corresponds to 
    the coefficients representing the n-qubit state |0...0>
    
    note: the state a0 |0...0> + a1 |0...1> + ... + a_{2^n - 1} |1...1>
    is stored as a list [a0, a1, ..., a_{2^n - 1}] 
    '''
    state = [(0.0 + 0.0j) for _ in range(2**num_qubits)]
    state[0] = 1.0 + 0.0j
    return state
    

def new_empty_state(num_qubits):
    '''
    returns a list of length 2^n initialized with zeros
    '''
    state = new_state(num_qubits)
    state[0] = 0.0 + 0.0j
    return state
    
    
#-----------------  TESTING  ----------------------------------------------------------------


def test_swap():
    # state = [0.2, 0.2+0.2j, 0.2, 0.2-0.2j, 0.2+0.2j, 0.2, 0.2, math.sqrt(0.6)]
    # state = [0, 1, 2, 3, 4, 5, 6, 7]
    num_qubits = 4
    state = [(0.0 + 0.0j) for _ in range(2**num_qubits)]
    state[0] = 0.5 + 0.0j
    state[1] = 0.0 + 0.5j
    state[12] = 0.5 + 0.0j
    state[13] = 0.0 + 0.5j

    '''
    compare this with http://www.davyw.com/quantum/
    step 1: the initial state can be created as follows:
    - set the number of bits to 4 and leave all qubits in state |0>
    -  put an H-gate on qubit 0 (top qubit),
       put an H-gate on qubit 3 (bottom qubit),
       put an S-gate on qubit 3 (bottom qubit),
       put a CNOT-gate between the 0th and 1st qubit (top qubit and the qubit under it),
       
    - Evaluate the circuit and you should get:
    0.50000000+0.00000000i	|0000>	25.0000%
    0.00000000+0.50000000i	|0001>	25.0000%
    0.50000000+0.00000000i	|1100>	25.0000%
    0.00000000+0.50000000i	|1101>	25.0000%

    step 2: put a swap gate between the top and bottom qubit
    - Evaluate the circuit and you should get:
    0.50000000+0.00000000i	|0000>	25.0000%
    0.50000000+0.00000000i	|0101>	25.0000%
    0.00000000+0.50000000i	|1000>	25.0000%
    0.00000000+0.50000000i	|1101>	25.0000%
    '''

    print("initial state:")
    print_state_with_basis_vectors(state)

    state = swap(state, 0, num_qubits-1)    # swap qubits in position 0 and num_qubits-1
    print("\n\n ---- new state after swap(0, 3):")
    print_state_with_basis_vectors(state)



def test_swap_2():
    num_qubits = 4
    state = new_empty_state(num_qubits)
    state[int('0100', 2)] = 1.0 + 0.0j     # corresponds to the state |0100>
    print("initial state:")
    print_state_with_basis_vectors(state)
    
    state = swap(state, 1, 3)
    print("\n\n ---- new state after swap(1, 3):")  # swap qubits in position 1 and 3
    print_state_with_basis_vectors(state)
    


#-----------------------------------------------------------------------------------------


def test_hadamard():
    num_qubits = 4
    state = [0 for _ in range(2**num_qubits)]
    
    # initialize state as |0000>
    state[0] = 1
    
    print("initial state:")
    print_state_with_basis_vectors(state)
    
    state = hadamard(state, 2)   # apply hadamard on qubit in position 2
    print("\n\n ---- new state after hadamard(state, 2):")
    print_state_with_basis_vectors(state)
    
    state = hadamard(state, 0)   # apply hadamard on qubit in position 0
    print("\n\n ---- new state after hadamard(state, 0):")
    print_state_with_basis_vectors(state)
    
    state = hadamard(state, 1)   # apply hadamard on qubit in position 1
    print("\n\n ---- new state after hadamard(state, 1):")
    print_state_with_basis_vectors(state)
    

#-----------------------------------------------------------------------------------------


def test_not_gate():
    num_qubits = 4
    state = new_state(num_qubits)
    
    print("initial state:")
    print_state_with_basis_vectors(state)
    
    state = hadamard(state, 1)   # apply hadamard on qubit in position 1
    print("\n\n ---- new state after hadamard(state, 1):")
    print_state_with_basis_vectors(state)
    
    state = not_gate(state, 2)    # apply not_gate on qubit in position 2
    print("\n\n ---- new state after not_gate(state, 2):")
    print_state_with_basis_vectors(state)


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


def test_controlled_not():
    num_qubits = 4
    state = new_state(num_qubits)
    
    print("initial state:")
    print_state_with_basis_vectors(state)
    
    state = hadamard(state, 1)   # apply hadamard on qubit in position 1
    print("\n\n ---- new state after hadamard(state, 1):")
    print_state_with_basis_vectors(state)
    
    state = hadamard(state, 2)    # apply not_gate on qubit in position 2
    print("\n\n ---- new state after not_gate(state, 2):")
    print_state_with_basis_vectors(state)

    state = controlled_not(state, 1, 3)  # control qubit is in position 1, 
                                         # target qubit is in position 3 on the right
    print("\n\n ---- new state after controlled_not(state, 1, 3):")
    print_state_with_basis_vectors(state)
     

     
def test_controlled_not_2():
    num_qubits = 4
    state = new_empty_state(num_qubits)
    state[2**num_qubits - 1] = 1.0 + 0j
    print(state)
    
    print("initial state:")
    print_state_with_basis_vectors(state)
    
    state = hadamard(state, 1)   # apply hadamard on qubit in position 1
    print("\n\n ---- new state after hadamard(state, 1):")
    print_state_with_basis_vectors(state)
    
    state = hadamard(state, 2)    # apply not_gate on qubit in position 2
    print("\n\n ---- new state after not_gate(state, 2):")
    print_state_with_basis_vectors(state)

    state = controlled_not(state, 1, 3)  # control qubit is in position 1, 
                                         # target qubit is in position 3 on the right
    print("\n\n ---- new state after controlled_not(state, 1, 3):")
    print_state_with_basis_vectors(state)



#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


if __name__ == '__main__':
    # test_swap()
    # test_swap_2()
    # test_hadamard()
    # test_not_gate()
    # test_controlled_not()
    test_controlled_not_2()
    
