import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

def print_banner(task_name):
    print(f"\n=== {task_name} ===")

def run_all_tasks():
    print_banner("THE LAZY ARTIST")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    print_banner("TASK 0: Creating Colored-MNIST")
    try:
        from task0_biased_dataset import ColoredMNIST, visualize_color_mapping
        easy_train = ColoredMNIST(root='./data', train=True, mode='easy')
        hard_test = ColoredMNIST(root='./data', train=False, mode='hard')
        print("Task 0 done")
    except Exception as e:
        print(f"Task 0 failed: {e}")
        return
    
    print_banner("TASK 1: Training Lazy CNN")
    try:
        from task1_cheater_model import train_lazy_model
        model, history = train_lazy_model(num_epochs=15)
        print(f"Task 1 done - Easy: {history['val_acc'][-1]:.1f}%, Hard: {history['hard_test_acc'][-1]:.1f}%")
    except Exception as e:
        print(f"Task 1 failed: {e}")
        return
    
    print_banner("TASK 2: Neuron Visualization")
    try:
        from task2_neuron_visualization import run_task2
        run_task2()
        print("Task 2 done")
    except Exception as e:
        print(f"Task 2 failed: {e}")
    
    print_banner("TASK 3: Grad-CAM")
    try:
        from task3_gradcam import run_task3
        run_task3()
        print("Task 3 done")
    except Exception as e:
        print(f"Task 3 failed: {e}")
    
    print_banner("TASK 4: Training Robust Models")
    try:
        from task4_intervention import run_task4
        results = run_task4()
        print("Task 4 done")
    except Exception as e:
        print(f"Task 4 failed: {e}")
    
    print_banner("TASK 5: Adversarial Attacks")
    try:
        from task5_adversarial import run_task5
        run_task5()
        print("Task 5 done")
    except Exception as e:
        print(f"Task 5 failed: {e}")
    
    print_banner("TASK 6: Sparse Autoencoders")
    try:
        from task6_sparse_autoencoders import run_task6
        run_task6()
        print("Task 6 done")
    except Exception as e:
        print(f"Task 6 failed: {e}")
    
    print("\nAll tasks complete.")

def run_single_task(task_num):
    tasks = {
        0: ('task0_biased_dataset', None),
        1: ('task1_cheater_model', 'train_lazy_model'),
        2: ('task2_neuron_visualization', 'run_task2'),
        3: ('task3_gradcam', 'run_task3'),
        4: ('task4_intervention', 'run_task4'),
        5: ('task5_adversarial', 'run_task5'),
        6: ('task6_sparse_autoencoders', 'run_task6'),
    }
    
    if task_num not in tasks:
        print(f"Invalid task number: {task_num}")
        print("Available tasks: 0-6")
        return
    
    module_name, func_name = tasks[task_num]
    
    try:
        module = __import__(module_name)
        if func_name:
            func = getattr(module, func_name)
            func()
    except Exception as e:
        print(f"Task {task_num} failed: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            task_num = int(sys.argv[1])
            run_single_task(task_num)
        except ValueError:
            if sys.argv[1] == 'all':
                run_all_tasks()
            else:
                print(f"Usage: python main.py [task_number|all]")
    else:
        print("Options: 0-6 or 'all'")
        choice = input("Choice: ").strip().lower()
        
        if choice == 'all':
            run_all_tasks()
        else:
            try:
                run_single_task(int(choice))
            except ValueError:
                print("Invalid choice")
