#include<bits/stdc++.h>
#include<iostream>
#include<vector>
#include<cmath>
using namespace std;

vector<vector<double>> transpose(vector<vector<double>> &temp)
{
    int n = temp.size();
    int m = temp[0].size();

    vector<vector<double>> T(m, vector<double>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            T[j][i] = temp[i][j];

    return T;
}

// Matrix Multiplication
vector<vector<double>> multiply(vector<vector<double>> &A, vector<vector<double>> &B)
{
    int n = A.size();
    int m = A[0].size();
    int p = B[0].size();

    vector<vector<double>> C(n, vector<double>(p, 0.0));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            for (int k = 0; k < m; k++)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}


// Gauss Jordan Matrix Inversion only for Non Singular Square Matrix

vector<vector<double>> inverse(vector<vector<double>> A)
{
    int n = A.size();
    vector<vector<double>> I(n, vector<double>(n, 0));

    // Create Identity Matrix
    for (int i = 0; i < n; i++)
        I[i][i] = 1;

    // Augmented Matrix [A | I]
    for (int i = 0; i < n; i++)
    {
        double diag = A[i][i];
        if (abs(diag) < 1e-9) {
            cerr << "Matrix is singular, cannot invert\n";
            exit(1);
        }

        for (int j = 0; j < n; j++) {
            A[i][j] /= diag;
            I[i][j] /= diag;
        }

        for (int k = 0; k < n; k++)
        {
            if (k == i) continue;
            double factor = A[k][i];

            for (int j = 0; j < n; j++) {
                A[k][j] -= factor * A[i][j];
                I[k][j] -= factor * I[i][j];
            }
        }
    }

    return I;
}



int main() 
{
    
    // Taking Inputs from CSV File
    // Open CSV file
    ifstream file("50_Startups.csv");
    if (!file.is_open()) {
        cerr << "Error: Could not open 50_Startups.csv\n";
        return 1;
    }

    vector<vector<double>> X; // Features: R&D, Admin, Marketing
    vector<double> y;         // Target: Profit

    string line;
    getline(file, line); // Skip header

    while (getline(file, line)) 
    {
        if (line.empty()) continue;

        stringstream ss(line);
        string rd, admin, marketing, state, profit;

        getline(ss, rd, ',');
        getline(ss, admin, ',');
        getline(ss, marketing, ',');
        getline(ss, state, ',');   // Ignored
        getline(ss, profit, ',');

        vector<double> temp;
        temp.push_back(stod(rd));
        temp.push_back(stod(admin));
        temp.push_back(stod(marketing));

        X.push_back(temp);
        y.push_back(stod(profit));
    }

    file.close();

    // Verify input
    cout << "Dataset Preview:\n";
    for (int i = 0; i < X.size(); i++) {
        cout << "R&D: " << X[i][0]
             << " | Admin: " << X[i][1]
             << " | Marketing: " << X[i][2]
             << " | Profit: " << y[i] << endl;
    }

    cout<<"Data Set Size : "<<X.size()<<endl;

    // Implementing OLS Logic

    int n = X.size();
    int dim = X[0].size();
    vector<double> y_pred(n); // For Later Use during Accuracy-Regression Metrics
    // Let's Calculate Weights Matrix for Finding the Best Fit Line
    //  Dimension of Weight Matrix is no.of independent Var+1 karon equation er subidharthe
    vector<double> weight(X[0].size()+1,1);
    weight[0] = 0;
    // We can perform Matrix Multiplication, Transpose but not inversion --> TC & Condition Issue so we need Gradient Descent for Calculation

    // Implement our Batch Gradient Descent, Stochastic Gradient Descent & our Mini Batch Gradient Descent
    int epochs = 100;
    double learning_rate = 0.01;    
    // IMPLEMENTING BATCH GRADIENT DESCENT 
           // ============ FEATURE SCALING (MANDATORY FOR GRADIENT DESCENT) ============
        cout << "\n=== Normalizing Features ===\n";
        vector<double> mean(dim, 0.0);
        vector<double> std_dev(dim, 0.0);

        // Calculate means
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < dim; k++) {
                mean[k] += X[j][k];
            }
        }
        for(int k = 0; k < dim; k++) {
            mean[k] /= n;
        }

        // Calculate standard deviations
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < dim; k++) {
                std_dev[k] += pow(X[j][k] - mean[k], 2);
            }
        }
        for(int k = 0; k < dim; k++) {
            std_dev[k] = sqrt(std_dev[k] / n);
            if(std_dev[k] < 1e-10) std_dev[k] = 1.0;
        }

        // Normalize X
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < dim; k++) {
                X[j][k] = (X[j][k] - mean[k]) / std_dev[k];
            }
        }

        // Normalize y
        double y_mean = 0.0;
        for(int j = 0; j < n; j++) {
            y_mean += y[j];
        }
        y_mean /= n;

        double y_std = 0.0;
        for(int j = 0; j < n; j++) {
            y_std += pow(y[j] - y_mean, 2);
        }
        y_std = sqrt(y_std / n);
        if(y_std < 1e-10) y_std = 1.0;

        for(int j = 0; j < n; j++) {
            y[j] = (y[j] - y_mean) / y_std;
        }

        cout << "Features normalized successfully!\n";

   
    for(int i=0;i<epochs;i++) // Epochs
    {
        // CALCULATING PREDICTED VALUE FOR ALL SAMPLES FIRST
        for(int j=0;j<n;j++) // Rows 
        {
            y_pred[j] = weight[0]; // Bias or INTERCEPT
            for(int k = 0; k < dim; k++) // Dimensions 
            {
                y_pred[j] += weight[k+1]*X[j][k];  
            }
        }

        // NOW UPDATE ALL WEIGHTS USING ALL SAMPLES (BATCH)

        // UPDATING INTERCEPT SEPARATELY
        double derivative = 2.0/n;  
        double val=0;
        for(int l = 0;l<n;l++)
        {
            val+=(y_pred[l]-y[l]);  
        }
        derivative*=val;
        weight[0] = weight[0] - learning_rate * derivative;

        // UPDATING OTHER WEIGHTS
        for(int k=1;k<=dim;k++)
        {
            derivative = 2.0/n;  
            double value=0;
            for(int l = 0;l<n;l++)
            {
                value+=(y_pred[l]-y[l])*X[l][k-1];  
            }
            derivative *= value;
            weight[k] = weight[k] - learning_rate * derivative;
        }
    }
    cout<<"The Best Fit Line is-"<<endl;
    cout<<"y = ";
    cout<<weight[0]<<" + ";
    for(auto &x:weight)
    {
        cout<<"("<<x<<")"<<"x + ";
    }

    return 0;
}
