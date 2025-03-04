This project is a simple comparison between brute-force (linear scan) and gradient descent for finding the best slope (m1) in a straight-line equation. The idea is to see which method works better in terms of accuracy and time efficiency.

**Why This Project?**
As part of a class task, I explored two different approaches to optimize a linear regression model and visualized their results. The aim was to understand:

How brute-force search works by scanning a range of values.
How gradient descent optimizes the slope step by step.
How both methods compare in terms of performance.

**Technologies Used**
Python (for implementation)
NumPy (for mathematical operations)
Matplotlib (for visualization)

**Project Structure**
📂 Project Folder  
│── main.py                 # Python script containing the implementation  
│── README.md               # Project documentation  
│── screenshots/            # Folder containing output graphs  


**How to Run the Code**
Install the required libraries (if not already installed):
In Cmd or Bash :
pip install numpy matplotlib
Run the script:
python main.py

**View the results:**
The script will first run a linear scan and display the corresponding graph.
Next, it will perform gradient descent and show its graph.
Finally, it will compare both methods in a single plot.
Screenshots of Output
(Screenshots of the output graphs should be added in the screenshots/ folder.)

**Results & Observations**
Linear scan takes a little longer but gives a fairly accurate result.
Gradient descent is faster but requires proper tuning of the learning rate.
Both methods give similar results, but gradient descent is more scalable for large datasets. Sample outputs are added below.

## **Screenshots of Output**

### **1. Linear Scan Output**
![Linear Scan](screenshots/linear_scan.png)

### **2. Gradient Descent Output**
![Gradient Descent](screenshots/gradient_descent.png)

### **3. Comparison Plot**
![Comparison](screenshots/comparison.png)

**Conclusion**
This project was a great way to understand two fundamental optimization techniques in machine learning. It also helped in improving my understanding of OOP in Python by structuring the code properly.

