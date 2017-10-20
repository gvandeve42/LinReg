import pandas as pd

weight1 = -1
bias = 10

test_inputs = [1, 4, 7, 20, 13, 5, 14, 3]
correct_outputs = [True, True, True, False, False, True, False, True]
outputs = []

for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input, linear_combination, output, is_correct_string])

num_wrong = len([output[2] for output in outputs if output[2] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', 'Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))
