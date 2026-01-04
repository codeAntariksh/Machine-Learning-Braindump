#include<bits/stdc++.h>
#include<iostream>
#include<vector>
#include<cmath>
using namespace std;
int main() 
{

    // Taking Inputs from CSV File
    ifstream file("placement.csv");

    if (!file.is_open()) {
        cerr << "Error: Could not open placement.csv\n";
        return 1;
    }

    vector<double> x; // cgpa
    vector<double> y; // package
    string line;
    
    // Skip header
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string cgpa, package;

        getline(ss, cgpa, ',');
        getline(ss, package, ',');

        x.push_back(stod(cgpa));
        y.push_back(stod(package));
    }

    file.close();





    // Verifying the Inputs
    // Verify
    for (int i = 0; i < x.size(); i++) {
        cout << "CGPA: " << x[i] << "  Package: " << y[i] << endl;
    }







    // Logic behind Simple Linear Regrssion
    
    // OLS Method
    
    // Taking a Pair for Easier Access and Reducing Time Complexity
    int n = x.size();
    vector<pair<double,double>> train;
    for(int i = 0; i<n;i++)
    {
        train.push_back({x[i],y[i]});
    }
    // Getting the Mean
    double x_mean = 0;
    double y_mean = 0;
    for(auto it: train)
    {
        x_mean += it.first;
        y_mean += it.second;
    }
    x_mean = x_mean / n;
    y_mean = y_mean / n;
    // Computing (x(i) - x_mean) * (y(i) - y_mean)
    double m;
    double num=0,den=0;
    for(auto val:train)
    {
        num += ((val.first - x_mean) * (val.second - y_mean));
        den += pow((val.first - x_mean),2); 
    }
    m = num / den;
    double b = y_mean - m * x_mean;
    cout<<"The Best Fit Line is: "<<"y = "<<m<<"*x + "<<b<<endl;
     
    // Regression Metrics
    cout<<"Regression Metrics:"<<endl;
     
    // Calculating MSE,MAE,RMSE
    
    // Calculating MSE
    // Taking Predicted Values over our Dataset with our m and b (Slope and Intercept)
    vector<double> y_pred;
    for(int i = 0;i<n;i++)
    {
        double value = m * x[i] + b;
        y_pred.push_back(value);
    }
    
    /// for MSE,MAE,RMSE
    double mse=0,mae=0,rmse=0;
    for(int i=0;i<n;i++)
    {
        double val = y[i] - y_pred[i];
        mse += (val*val);
        mae+=abs(val);
    }
    rmse = mse;
    mae/=n;
    mse/=n;
    rmse = pow(rmse/n,0.5);
    
    cout<<"Mean Absolute Error: "<<mae<<endl;
    cout<<"Mean Squared Error: "<<mse<<endl;
    cout<<"Root Mean Squared Error: "<<rmse<<endl;

    
    
    
    
    return 0;
}