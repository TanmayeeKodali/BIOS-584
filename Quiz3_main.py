from self_py_fun.Quiz3Fun import *

# You can use this .py script to perform debugging task.
sample_arr_1 = np.array([1,2,3,4,5])
sample_arr_2 = np.sin(np.arange(0, 2.1*np.pi, np.pi/10))

# Testing incorrect function
d_1_wrong = compute_D_partial(sample_arr_1)
print(f"Incorrect D (compute_D_partial): {d_1_wrong:.2f}")  # Should print 1.85

# Testing correct function (D_correct)
d_val_1 = compute_D_correct(sample_arr_1)
print(f"d_val_1: {d_val_1:.2f}")  # Should print 5.66

d_val_2 = compute_D_correct(sample_arr_2)
print(f"d_val_2: {d_val_2:.2f}")