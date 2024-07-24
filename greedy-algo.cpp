#include <bits/stdc++.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>
#include <vector>
using namespace std;
typedef vector<int> VI;
typedef vector<vector<int>> VVI;
typedef vector<pair<int, int>> VPI;
typedef vector<string> VS;
typedef queue<int> QU;
typedef queue<pair<int, int>> QP;
typedef queue<pair<pair<int, int>, int>> QPP;
#define PB push_back
#define SZA(arr) (sizeof(arr) / sizeof(arr[0]))
#define SZ(x) ((int)x.size())
#define LEN(x) ((int)x.length())
#define REV(x) reverse(x.begin(), x.end());
#define trav(a, x) for (auto &a : x)
#define FOR(i, n) for (int i = 0; i < n; i++)
#define FOR_INNER(j, i, n) for (int j = i; j < n; j++)
#define FOR1(i, n) for (int i = 1; i <= n; i++)
#define SORT(x) sort(x.begin(), x.end())
// Short function start-->>
void printArray(int arr[], int length)
{
    for (int i = 0; i < length; ++i)
    {
        cout << arr[i] << " ";
    }
}
void printVector(vector<int> &arr)
{
    for (auto it : arr)
    {
        cout << it << " ";
    }
}
void printVectorString(vector<string> &arr)
{
    for (auto it : arr)
    {
        cout << it << endl;
    }
}
void printVectorVector(vector<vector<int>> x)
{
    for (const auto &row : x)
    {
        cout << "[";
        for (const auto &element : row)
        {
            cout << element << " ";
        }
        cout << "]";
        cout << std::endl;
    }
}
void printVectorVectorString(vector<vector<string>> x)
{
    for (const auto &row : x)
    {
        cout << "[";
        for (const auto &element : row)
        {
            cout << element << " ";
        }
        cout << "]";
        cout << std::endl;
    }
}
void printString(string s, int length)
{
    for (int i = 0; i < length; ++i)
    {
        cout << s[i] << " ";
    }
}
void printStack(stack<string> s)
{
    while (!s.empty())
    {
        cout << s.top() << " ";
        s.pop();
    }
    cout << endl;
}

// Short function end-->>
/*

Li'll Interoduction----->>>
1️⃣ what is Greedy Algorithm?
-->>Greedy is an algorithmic paradigm that builds up a solution piece by piece, always choosing the next piece that offers the most obvious and immediate benefit. So the problems where choosing locally optimal also leads to global solution are the best fit for Greedy.
2️⃣ Standard Greedy Algorithms:
-->
Activity Selection Problem
Job Sequencing Problem
Huffman Coding
Huffman Decoding
Water Connection Problem
Minimum Swaps for Bracket Balancing
Egyptian Fraction
Policemen catch thieves
Fitting Shelves Problem
Assign Mice to Holes
3️⃣
4️⃣
*/

/*
1. Assign Cookies
ANS : Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.

Each child i has a greed factor g[i], which is the minimum size of a cookie that the child will be content with;
and each cookie j has a size s[j]. If s[j] >= g[i], we can assign the cookie j to the child i,
and the child i will be content. Your goal is to maximize the number of your content children and
output the maximum number.
Input : g = [1,2], s = [1,2,3]  || Output :2
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC : O(max(m, n) + (m * log m) + (n * log n)), where ,mlogm and nlogn is for sorting
// SC : O(1)
// Find minimum sized cookie which satisfies the child
/*
Intuitions : We've to give mazimum cookies so if you sort both cookies and children array then loop simultaneously
and assign cookies to children if chd[i]<=cook[i] else go to next cookie Base condition is if any one loop exceed then return loop index
*/
int findContentChildren(VI &child, VI &cookie)
{
    int m = SZ(child);
    int n = SZ(cookie);
    SORT(child);
    SORT(cookie);
    int i = 0, j = 0;
    while (i < m && j < n)
    {
        if (child[i] <= cookie[j])
        { // satisfied
            i++;
        }
        j++;
    }
    return i;
}

/*
2. Fractional Knapsack
ANS : Given weights and values of N items, we need to put these items in a knapsack of capacity W to get the maximum total value in the knapsack.
Note: Unlike 0/1 knapsack, you are allowed to break the item here.
Input :  N = 3, W = 50 value[] = {60,100,120} weight[] = {10,20,30} || Output : 240.000000
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC : O(n log n + n). O(n log n) to sort the items and O(n) to iterate through all the items for calculating the answer.
// SC : O(1)
/*
Approach: The greedy method to maximize our answer will be to pick up the items with higher values.
Since it is possible to break the items as well we should focus on picking up items having higher value /weight first.
To achieve this, items should be sorted in decreasing order with respect to their value /weight.
Once the items are sorted we can iterate. Pick up items with weight lesser than or equal to the current capacity of the knapsack.
In the end, if the weight of an item becomes more than what we can carry, break the item into smaller units.
Calculate its value according to our current capacity and add this new value to our answer.

Explain : Sort according to desc order(3,2,1) in value/weight cz you can pick elem that not exceed knapsack weight
so you picked up and fill.
Here, is a other word you can use fraction so you can take any weights fraction so it will also descrease value

*/
// #### Its diffrent from the DP cz here you can break the items so thats why its called  Fractional Knapsack
struct Item
{
    int value;
    int weight;
};
class Solution
{
public:
    bool static comp(Item a, Item b) // for pair<int,int>a
    {

        double r1 = (double)a.value / (double)a.weight; // here you can use instea of weight=first and value=second
        double r2 = (double)b.value / (double)b.weight;
        return r1 > r2;
    }
    // function to return fractionalweights
    double fractionalKnapsack(int W, Item arr[], int n)
    {

        sort(arr, arr + n, comp);
        // if you have VPI then you can use a.begin(),a.end(),comp

        int curWeight = 0;
        double finalvalue = 0.0;

        for (int i = 0; i < n; i++)
        {

            if (curWeight + arr[i].weight <= W)
            { // If this weight fit then add weight and value
                curWeight += arr[i].weight;
                finalvalue += arr[i].value;
            }
            else
            { // If not fit then take fraction of it
                int remain = W - curWeight;
                finalvalue += (arr[i].value / (double)arr[i].weight) * (double)remain;
                break;
            }
        }

        return finalvalue;
    }
};
/*
3. Find minimum number of coins
ANS :  Given a value V, if we want to make a change for V Rs, and we have an infinite supply of each of the denominations in Indian currency, i.e., we have an infinite supply of { 1, 2, 5, 10, 20, 50, 100, 500, 1000} valued coins/notes, what is the minimum number of coins and/or notes needed to make the change.
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC :
// SC :
// We will keep a pointer at the end of the array i. Now while(V >= coins[i]) we will reduce V by coins[i] and add it to the ans array.
// We will also ignore the coins which are greater than V and the coins which are less than V. We consider them and reduce the value of V by coins[I].
int minCoins(int coins[], int n, int V)
{
    // base case
    if (V == 0)
        return 0;
    int cnt = 0;
    for (int i = n - 1; i >= 0; i--)
    {
        while (V >= coins[i])
        {
            V -= coins[i];
            cnt++;
        }
    }
    return cnt;
}
/*
4. Lemonade Change
ANS : At a lemonade stand, each lemonade costs $5. Customers are standing in a queue to buy from you and order one at a time (in the order specified by bills). Each customer will only buy one lemonade and pay with either a $5, $10, or $20 bill. You must provide the correct change to each customer so that the net transaction is that the customer pays $5.
Note that you do not have any change in hand at first.
Given an integer array bills where bills[i] is the bill the ith customer pays, return true if you can provide every customer with the correct change, or false otherwise.
Input :  [5, 5, 5, 10, 20] || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC :O(n)
// SC :O(1)
/*
Intuitions : If seller have already [5, 5, 5, 10, 20] this coins at first he can sell 3 lemonade so
you have now 5+5+5 now second if he sell lemonade he can give 5$ change also same for third he can give 15$ return.
So, I'll be carrying a variable that keeps a track of 5 & 10 & 20 anytime a customer comes
with a 5 denomination i simply add it the 5 variable
with a 10 denomination i add it to 10 and reduce from 5
with a 20 denomination i add it to 20 and reduce from 5 & 10 || (5,10),(5,5,5)
we're returning 5 & 10 so i don't need 20 to remember
*/
bool lemonadeChange(vector<int> &bills)
{
    int count5 = 0, count10 = 0;
    trav(bill, bills)
    {
        if (bill == 5)
        {
            count5++;
        }
        else if (bill == 10)
        {
            if (count5 == 0)
            {
                return false; // Cannot provide change
            }
            count5--;
            count10++;
        }
        else
        { // bill == 20
            if (count10 > 0 && count5 > 0)
            {
                count10--;
                count5--;
            }
            else if (count5 >= 3)
            {
                count5 -= 3;
            }
            else
            {
                return false; // Cannot provide change
            }
        }
    }
    return true; // All customers provided with correct change
}
/*
5. Valid Parenthesis String
ANS : Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.
The following rules define a valid string:
Any left parenthesis '(' must have a corresponding right parenthesis ')'.
Any right parenthesis ')' must have a corresponding left parenthesis '('.
Left parenthesis '(' must go before the corresponding right parenthesis ')'.
'*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".
Input : s = "(*))"  || Output : true
*/
// Bruteforce ----------->
// TC :
// SC :
//  Recursion
// Better ----------->
// TC :
// SC :
// Memoization
// Optimal ---------->
// TC : O(n)
// SC :O(1)
bool checkValidString(string s)
{
    int minOpen = 0, maxOpen = 0;
    for (char ch : s)
    {
        if (ch == '(')
        {
            minOpen++;
            maxOpen++;
        }
        else if (ch == ')')
        {
            minOpen = max(minOpen - 1, 0);
            maxOpen--;
        }
        else
        { // '*'
            minOpen = max(minOpen - 1, 0);
            maxOpen++;
        }
        if (maxOpen < 0)
        {
            return false;
        }
    }
    return minOpen == 0;
}
/*
6. N meetings in one room
ANS : You are given the schedule of 'N' meetings with their start time 'Start[i]' and end time 'End[i]'.
You have only 1 meeting room. So, you need to return the maximum number of meetings you can organize.
Note:
The start time of one chosen meeting can’t be equal to the end time of the other chosen meeting.
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC : O(N)+O(NlogN)+O(N)
// The result is dominated by the term with the highest growth rate as n increases. In this case, O(nlogn) grows faster than O(n)
// for large values of n.
// When combining the terms, the lower-order terms (O(n)) become insignificant in the presence of the higher-order term
// (O(nlogn)). Hence, the overall time complexity simplifies to: O(n)+O(nlogn)+O(n)=O(nlogn)
// SC :O(N)
struct meeting
{
    int start;
    int end;
    int pos;
};

class Nmeet
{
public:
    bool static comparator(struct meeting m1, meeting m2)
    {
        if (m1.end < m2.end)
            return true;
        else if (m1.end > m2.end)
            return false;
        else if (m1.pos < m2.pos)
            return true;
        return false;
    }
    int maximumMeetings(vector<int> &start, vector<int> &end)
    {
        int n = start.size();
        vector<meeting> meet(n);
        // Also instead you can use VPI same TC,SC(without object)
        // VPI meet(n);
        // for (int i = 0; i < n; ++i)
        // {
        //     meetings[i] = {end[i], start[i]};
        // }

        // Initialize meetings
        for (int i = 0; i < n; i++)
        {
            meet[i].start = start[i];
            meet[i].end = end[i];
            meet[i].pos = i + 1;
        }

        // Sort meetings based on end times
        sort(meet.begin(), meet.end(), comparator);
        cout << "Sorted meetings:" << endl;
        for (const auto &m : meet)
        {
            cout << "Meeting " << m.pos << ": Start = " << m.start << ", End = " << m.end << endl;
        }

        // Now compare meetings end to next meetings start as we sorted according to start and end
        int limit = meet[0].end;
        int count = 1; // Count of meetings
        for (int i = 1; i < n; i++)
        {
            if (meet[i].start > limit)
            {
                limit = meet[i].end;
                count++;
            }
        }

        return count;
    }
};

/*
7. Jump Game
ANS : You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.
Return true if you can reach the last index, or false otherwise.
Input : nums = [2,3,1,1,4]  || Output : true
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC :O(N)
// SC :O(1)
/*
Intuition : Array elems represents your maximum jump length if its 2 then you can got next 1 elem or next 2 elem
As our observation if we can manage to max Index that can be touched and max index>=n then we can say Yes
we can reach the end so try a loop from 0->n and add index+arr[i] this way you can touched max index
Here you've to add 2 condition like if my current index is less than max index then return false and 2 nd condition is
if we exceed array length or reached we can return yes
*/
bool canJump(vector<int> &nums)
{
    int furthestReachable = 0;
    int n = SZ(nums);
    FOR(i, n)
    {
        // Update furthest reachable index based on current element
        if (i > furthestReachable) // If you have't reached me that means you can't jump from here
        {
            return false; // If current position is unreachable, return false
        }
        furthestReachable = max(furthestReachable, i + nums[i]);

        // If furthest reachable index surpasses or reaches end of array, return true
        if (furthestReachable >= n - 1)
        {
            return true;
        }
    }
    return false;
}
/*
8. Jump Game II
ANS : You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].
Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where:
0 <= j <= nums[i] and
i + j < n
Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].
Input : nums = [2,3,1,1,4]  || Output :2
*/
// Bruteforce ----------->

// TC : Exponential in nature it can be O(N^n)
// SC : Auxiliary stack space O(n)
/* Intuitions : Here confirmly states that we can reach last index but we have to retutn total no of jumps.
so we're trying using recursive version so we can pick minimum
*/
int jumpHelper(int ind, int jump, VI &nums, int n)
{
    if (ind >= n - 1)
        return jump;
    int mini = INT_MAX;
    for (int i = 1; i <= nums[ind]; i++)
    {
        mini = min(mini, jumpHelper(ind + i, jump + 1, nums, n));
    }
    return mini;
}
int jumpRecr(vector<int> &nums)
{
    int n = SZ(nums);
    return jumpHelper(0, 0, nums, n);
}
// Better ------using Memo----->
// TC : O(N^2)
// SC :O(N)
int jumpMemoHelper(int ind, vector<int> &nums, vector<int> &memo)
{
    int n = nums.size();
    if (ind >= n - 1)
        return 0; // If the current index is at or beyond the last index
    if (memo[ind] != -1)
        return memo[ind]; // Check if result is already computed

    int mini = INT_MAX;
    for (int i = 1; i <= nums[ind]; i++)
    { // Include nums[ind] as a valid jump
        if (ind + i < n)
        { // Ensure we do not go out of bounds
            int jumps = jumpMemoHelper(ind + i, nums, memo);
            if (jumps != INT_MAX)
            {                                // Check if a valid path was found
                mini = min(mini, jumps + 1); // Increment jump count and find the minimum
            }
        }
    }
    return memo[ind] = mini; // Store the result in the memo array
}
int jumpMemo(vector<int> &nums)
{
    int n = SZ(nums);
    vector<int> memo(n, -1); // Initialize memo array with -1
    return jumpMemoHelper(0, nums, memo);
}
// Optimal ---------->
// TC :O(N)
// SC :O(1)
/* Intuitions : Here confirmly states that we can reach last index but we have to retutn total no of jumps.
so
*/
int jump(vector<int> &nums)
{
    int n = nums.size();
    if (n <= 1)
        return 0;
    int jumps = 0;
    int furthestReachable = 0;
    int currentMaxReachable = 0;
    for (int i = 0; i < n - 1; ++i)
    {
        currentMaxReachable = max(currentMaxReachable, i + nums[i]);
        if (i == furthestReachable)
        {
            jumps++;
            furthestReachable = currentMaxReachable;
        }
    }
    return jumps;
}
/*
9. Minimum Platforms
ANS : Given arrival and departure times of all trains that reach a railway station. Find the minimum number of platforms required for the railway station so that no train is kept waiting.
Consider that all the trains arrive on the same day and leave on the same day. Arrival and departure time can never be the same for a train but we can have arrival time of one train equal to departure time of the other. At any given instance of time, same platform can not be used for both departure of a train and arrival of another train. In such cases, we need different platforms.
Input :   n = 6
arr[] = {0900, 0940, 0950, 1100, 1500, 1800}
dep[] = {0910, 1200, 1120, 1130, 1900, 2000} || Output : 3
*/
// Bruteforce ----------->
// TC :  O(n^2)  (due to two nested loops).
// SC :O(1)
int countPlatformsBruteforce(int n, int arr[], int dep[])
{
    int ans = 1; // final value
    for (int i = 0; i <= n - 1; i++)
    {
        int count = 1; // count of overlapping interval of only this   iteration
        for (int j = i + 1; j <= n - 1; j++)
        {
            if ((arr[i] >= arr[j] && arr[i] <= dep[j]) ||
                (arr[j] >= arr[i] && arr[j] <= dep[i]))
            {
                count++;
            }
        }
        ans = max(ans, count); // updating the value
    }
    return ans;
}
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC : O(nlogn) Sorting takes O(nlogn) and traversal of arrays takes O(n) so overall time complexity is O(nlogn).
// SC : O(1)
int countPlatformsOptimal(int n, int arr[], int dep[])
{
    sort(arr, arr + n);
    sort(dep, dep + n);

    int ans = 1;
    int count = 1;
    int i = 1, j = 0;
    while (i < n && j < n)
    {
        if (arr[i] <= dep[j]) // one more platform needed
        {
            count++;
            i++;
        }
        else // one platform can be reduced
        {
            count--;
            j++;
        }
        ans = max(ans, count); // updating the value with the current maximum
    }
    return ans;
}
/*
10. Job sequencing Problem
ANS : You are given a 'Nx3' 2-D array 'Jobs' describing 'N' jobs where 'Jobs[i][0]' denotes the id of 'i-th' job, 'Jobs[i][1]' denotes the deadline of 'i-th' job, and 'Jobs[i][2]' denotes the profit associated with 'i-th job'.
You will make a particular profit if you complete the job within the deadline associated with it. Each job takes 1 unit of time to be completed, and you can schedule only one job at a particular time.
Return the number of jobs to be done to get maximum profit.
Note :
If a particular job has a deadline 'x', it means that it needs to be completed at any time before 'x'.
Assume that the start time is 0.
Input :  'N' = 3, Jobs = [[1, 1, 30], [2, 3, 40], [3, 2, 10]] || Output : [3 80]
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC :  O(N log N) + O(N*M).
// SC : O(M) for an array that keeps track on which day which job is performed if M is the maximum deadline available.
/*
Intuition : First sort it by job profits to find max profit. Then we need to assign a slot array which have max deadline+1. In this slot assign jobs so for that you have to traverse the through jobs and Check for index to place its from right->left checking for slots for right to left you again need a loop from deadline to 1 not 0. If you find a slot then assign job id to slot and increment cnt Jobs also profit of this jobs.

 Why Slot 0 is Not Used:
The job deadlines start from 1, so slot 0 is not needed and is left unused.
We use slot array indexing starting from 1 to match job deadlines directly.
*/
struct Job
{
    int id;     // Job Id
    int dead;   // Deadline of job
    int profit; // Profit if job is over before or on deadline
};
class J
{
public:
    bool static comparison(Job a, Job b)
    {
        return (a.profit > b.profit);
    }
    // Function to find the maximum profit and the number of jobs done
    pair<int, int> JobScheduling(Job arr[], int n)
    {

        // sort it by profit
        sort(arr, arr + n, comparison);
        // Find max deadline
        int maxi = arr[0].dead;
        for (int i = 1; i < n; i++)
        {
            maxi = max(maxi, arr[i].dead);
        }
        int slot[maxi + 1];
        for (int i = 0; i <= maxi; i++)
            slot[i] = -1; // Assign all index with -1

        // Iterate through the jobs and assign them to slots
        int cntJobs = 0, jobProfit = 0;
        FOR(i, n)
        { // Pick the job
            for (int j = arr[i].dead; j > 0; j--)
            { // Check for index to place its from right->left checking for slots
                if (slot[j] == -1)
                {
                    slot[j] = 1;
                    cntJobs++;
                    jobProfit += arr[i].profit;
                    break; // If job is placed then not need go to the left slots
                }
            }
        }

        return make_pair(cntJobs, jobProfit);
    }
};
// Another type
bool comparison(const vector<int> &a, const vector<int> &b)
{
    return a[2] > b[2]; // Sort by profit in non-increasing order
}

vector<int> jobScheduling(vector<vector<int>> &jobs)
{
    // Step 1: Sort the jobs by profit in non-increasing order
    sort(jobs.begin(), jobs.end(), comparison);
    printVectorVector(jobs);
    cout << endl;

    // Step 2: Initialize an array to track slots
    int maxDeadline = 0;
    for (const auto &job : jobs)
    {
        maxDeadline = max(maxDeadline, job[1]);
    }
    vector<int> slot(maxDeadline + 1, -1); // Initialize slots with -1 indicating empty

    // Step 3: Iterate through the sorted jobs and assign them to slots
    int countJobs = 0, jobProfit = 0;
    for (const auto &job : jobs)
    {
        int deadline = job[1];
        for (int i = deadline; i > 0; --i)
        {
            if (slot[i] == -1)
            {
                slot[i] = job[0]; // Assign job id to the slot
                countJobs++;
                jobProfit += job[2];
                break;
            }
        }
    }

    // Return the count of jobs scheduled
    return {countJobs, jobProfit};
}
/*
11. Candy
ANS : There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.

You are giving candies to these children subjected to the following requirements:

Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
Return the minimum number of candies you need to have to distribute the candies to the children.
Input :  ratings = [1,0,2] || Output : 5 [2,1,2]
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC : O(N)
// SC :O(N)
int candy(vector<int> &ratings)
{
    int n = ratings.size();
    int totalCandies = 0;

    // Initialize the candies array to store the candies assigned to each child
    vector<int> candies(n, 1);

    // First pass: Ensure higher rated child gets more candies than its left neighbor
    for (int i = 1; i < n; ++i)
    {
        if (ratings[i] > ratings[i - 1])
        {
            candies[i] = candies[i - 1] + 1;
        }
    }

    // Second pass: Update candies count for the right neighbors
    for (int i = n - 2; i >= 0; --i)
    {
        if (ratings[i] > ratings[i + 1] && candies[i] <= candies[i + 1])
        {
            candies[i] = candies[i + 1] + 1;
        }
    }

    // Sum up the total candies distributed
    for (int candyCount : candies)
    {
        totalCandies += candyCount;
    }

    return totalCandies;
}
/*
12. SJF (shortest job first)
ANS : You have to implement the shortest job first scheduling algorithm.
Shortest Job First is an algorithm in which the process having the smallest execution(burst) time is chosen for the next execution. Here, you will implement a non - preemptive version (a process will wait till process(es) with shorter burst time executes). You have to return the average waiting for the given number of processes.
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// Time Complexity: O(N ^ 2)
// Space complexity: O(N)

float sjfBetter(int n, vector<int> &arrivalTime, vector<int> &burstTime)
{
    vector<int> completionTime(n);
    vector<int> waitingTime(n);
    vector<int> completed(n);

    int systemTime = 0, totalProcesses = 0;
    float avgwaitingTime = 0;

    // Initializing all process as undone.
    for (int i = 0; i < n; i++)
    {
        completed[i] = 0;
    }

    // Till all the processes are done.
    while (totalProcesses != n)
    {
        int check = n, min = INT_MAX;

        for (int i = 0; i < n; i++)
        {
            // If the process arrival time is less than system time and it is not completed
            // and burstTime is smallest of this process this process will be executed first.
            if ((arrivalTime[i] <= systemTime) and (completed[i] == 0) and (burstTime[i] < min))
            {
                min = burstTime[i];
                check = i;
            }
        }

        // No process in the queue.
        if (check == n)
        {
            systemTime++;
        }
        else
        {
            completionTime[check] = systemTime + burstTime[check];
            systemTime += burstTime[check];
            waitingTime[check] = completionTime[check] - arrivalTime[check] - burstTime[check];
            completed[check] = 1;
            totalProcesses++;
        }
    }

    // Sum for calculating averages.
    for (int i = 0; i < n; i++)
    {
        avgwaitingTime += waitingTime[i];
    }

    float ans;

    ans = (float)(avgwaitingTime / n);
    return ans;
}
// Optimal ---------->
// TC : O(NlogN)
// SC :O(N)
struct Process
{
    int id;          // Process ID
    int arrivalTime; // Arrival time of the process
    int burstTime;   // Burst time of the process
};

// Custom comparison function for priority queue based on burst time
struct CompareBurstTime
{
    bool operator()(const Process &a, const Process &b)
    {
        return a.burstTime > b.burstTime;
    }
};

float sjf(int n, vector<int> &arrivalTime, vector<int> &burstTime)
{
    vector<Process> processes(n);

    // Create a vector of processes
    for (int i = 0; i < n; ++i)
    {
        processes[i] = {i + 1, arrivalTime[i], burstTime[i]};
    }

    // Sort processes based on arrival time
    sort(processes.begin(), processes.end(), [](const Process &a, const Process &b)
         { return a.arrivalTime < b.arrivalTime; });

    priority_queue<Process, vector<Process>, CompareBurstTime> pq; // Priority queue based on burst time
    int currentTime = 0;
    float totalWaitingTime = 0;

    int i = 0;
    while (i < n || !pq.empty())
    {
        if (pq.empty())
        {
            currentTime = max(currentTime, processes[i].arrivalTime);
        }

        // Push all processes that have arrived by the current time
        while (i < n && processes[i].arrivalTime <= currentTime)
        {
            pq.push(processes[i]);
            i++;
        }

        // If there are processes in the queue, execute the shortest one
        if (!pq.empty())
        {
            Process currentProcess = pq.top();
            pq.pop();
            totalWaitingTime += currentTime - currentProcess.arrivalTime;
            currentTime += currentProcess.burstTime;
        }
    }

    // Return average waiting time
    return totalWaitingTime / n;
}
/*
Intuition : As per SJF algorithm it is starting that the one with the least execution time
will be executed first so as per input after sorting [1,2,3,4,7]
for 1 it stared at time 0->1=0
for 2 it stared at time 1->1+2=1
for 3 it stared at time 3->1+2+3=3
for 4 it stared at time 6->1+2+3+4=6
for 7 it stared at time 10->1+2+3+4+7=10
now sum up the wating time and calculate avarage like sum=(0+1+3+6+10)/5=4
So, First sort the array
then init variable timer and waitTime
then loop from 0->n-1
then waitTime +=time and time +=arr[i]
and return waitTime/n
*/
// Time : O(n)+O(nlogn)
long long solveSjf(vector<int> &bt)
{
    int n = SZ(bt);
    SORT(bt);
    long long time = 0, waitTime = 0;
    FOR(i, n)
    {
        waitTime += time;
        time += bt[i];
    }
    return (waitTime / n);
}
/*
13. LRU Page Replacement Algo
ANS : LRU stands for Least Recently Used. As the name suggests, this algorithm is based on the strategy that whenever a page fault occurs, the least recently used page will be replaced with a new page.
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC :O(N)
// SC :O(capacity)
int pageFaults(int pages[], int n, int capacity)
{
    unordered_map<int, list<int>::iterator> cache;
    list<int> recentlyUsed;
    int faults = 0;

    for (int i = 0; i < n; ++i)
    {
        // If page is not found in cache
        if (cache.find(pages[i]) == cache.end())
        {
            faults++;

            // If cache is at full capacity, evict least recently used page
            if (cache.size() == capacity)
            {
                int leastRecentlyUsed = recentlyUsed.back();
                recentlyUsed.pop_back();
                cache.erase(leastRecentlyUsed);
            }

            // Add page to cache
            recentlyUsed.push_front(pages[i]);
            cache[pages[i]] = recentlyUsed.begin();
        }
        else
        {
            // Update recently used list
            recentlyUsed.erase(cache[pages[i]]);
            recentlyUsed.push_front(pages[i]);
            cache[pages[i]] = recentlyUsed.begin();
        }
    }

    return faults;
}
/*
14. Insert Interval
ANS : You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC :O(N)
// SC :O(N)

vector<vector<int>> insert(vector<vector<int>> &intervals, vector<int> &newInterval)
{
    vector<vector<int>> result;

    int n = intervals.size();
    int i = 0;

    // Add all intervals that come before the newInterval
    while (i < n && intervals[i][1] < newInterval[0])
    {
        result.push_back(intervals[i]);
        i++;
    }

    // Merge intervals that overlap with the newInterval
    while (i < n && intervals[i][0] <= newInterval[1])
    {
        newInterval[0] = min(newInterval[0], intervals[i][0]);
        newInterval[1] = max(newInterval[1], intervals[i][1]);
        i++;
    }

    result.push_back(newInterval);

    // Add remaining intervals after the merged interval
    while (i < n)
    {
        result.push_back(intervals[i]);
        i++;
    }

    return result;
}

/*
15. Merge Intervals
ANS : Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
Input :  intervals = [[1,3],[2,6],[8,10],[15,18]] || Output : [[1,6],[8,10],[15,18]]
*/
// Bruteforce ----------->
// TC : O(N*logN) + O(2*N)
// SC : O(N)
vector<vector<int>> mergeOverlappingIntervalsBruteforce(vector<vector<int>> &arr)
{
    int n = arr.size(); // size of the array

    // sort the given intervals:
    sort(arr.begin(), arr.end());

    vector<vector<int>> ans;

    for (int i = 0; i < n; i++)
    { // select an interval:
        int start = arr[i][0];
        int end = arr[i][1];

        // Skip all the merged intervals:
        if (!ans.empty() && end <= ans.back()[1])
        {
            continue;
        }

        // check the rest of the intervals:
        for (int j = i + 1; j < n; j++)
        {
            if (arr[j][0] <= end)
            {
                end = max(end, arr[j][1]);
            }
            else
            {
                break;
            }
        }
        ans.push_back({start, end});
    }
    return ans;
}
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC : O(N*logN) + O(N)
// SC :O(N)
vector<vector<int>> mergeOverlappingIntervalsOptimal(vector<vector<int>> &arr)
{
    int n = arr.size(); // size of the array

    // sort the given intervals:
    sort(arr.begin(), arr.end());

    vector<vector<int>> ans;

    for (int i = 0; i < n; i++)
    {
        // if the current interval does not
        // lie in the last interval:
        if (ans.empty() || arr[i][0] > ans.back()[1])
        {
            ans.push_back(arr[i]);
        }
        // if the current interval
        // lies in the last interval:
        else
        {
            ans.back()[1] = max(ans.back()[1], arr[i][1]);
        }
    }
    return ans;
}
/*
16. Non-overlapping Intervals
ANS : Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC :O(NlogN)
// SC :O(1)
int eraseOverlapIntervals(vector<vector<int>> &intervals)
{
    if (intervals.empty())
        return 0;

    // Sort intervals by ending points
    sort(intervals.begin(), intervals.end(), [](const vector<int> &a, const vector<int> &b)
         { return a[1] < b[1]; });

    int end = intervals[0][1]; // Ending point of the first interval
    int removeCount = 0;

    // Iterate through sorted intervals
    for (int i = 1; i < intervals.size(); ++i)
    {
        // If current interval overlaps with the previous one, remove it
        if (intervals[i][0] < end)
        {
            removeCount++;
        }
        else
        {
            end = intervals[i][1]; // Update ending point
        }
    }

    return removeCount;
}

// ===============================SOME IMPORTANT ALGORITHMS===============================>>
/*
1. Activity Selection Problem
You are given n activities with their start and finish times. Select the maximum number of activities that can be performed by a single person, assuming that a person can only work on a single activity at a time.
Input: start[]  =  {10, 12, 20}, finish[] =  {20, 25, 30}
Output: 0 2
TC : O(N)
SC : O(1)
*/
// bool activityCompare(Activitiy s1, Activitiy s2)
// {
//     return (s1.finish < s2.finish);
// }
void printMaxActivities(int s[], int f[], int n)
{
    // if not sorted then sort according to finish time
    // sort(arr, arr + n, activityCompare);// here array will contain start and finish time, then time will O(logN)
    int i, j;

    cout << "Following activities are selected" << endl;

    // The first activity always gets selected
    i = 0;
    cout << i << " ";

    // Consider rest of the activities
    for (j = 1; j < n; j++)
    {
        // If this activity has start time greater than or
        // equal to the finish time of previously selected
        // activity, then select it
        if (s[j] >= f[i])
        {
            cout << j << " ";
            i = j;
        }
    }
}

/*
2. Job Sequencing Problem
You can found up.
*/

/*
3. Huffman Coding
Its from tree question
*/

/*
3. Huffman Decoding
Its from tree question
*/

/*
4. Water connection problem
Every house in the colony has at most one pipe going into it and at most one pipe going out of it. Tanks and taps are to be installed in a manner such that every house with one outgoing pipe but no incoming pipe gets a tank installed on its roof and every house with only an incoming pipe and no outgoing pipe gets a tap.
Given two integers n and p denoting the number of houses and the number of pipes. The connections of pipe among the houses contain three input values: a_i, b_i, d_i denoting the pipe of diameter d_i from house a_i to house b_i, find out the efficient solution for the network.
The output will contain the number of pairs of tanks and taps t installed in first line and the next t lines contain three integers: house number of tank, house number of tap and the minimum diameter of pipe between them.
TC : O(N)
SC : O(N)
*/
// number of houses and number
// of pipes
int number_of_houses, number_of_pipes;

// Array rd stores the
// ending vertex of pipe
int ending_vertex_of_pipes[1100];

// Array wd stores the value
// of diameters between two pipes
int diameter_between_two_pipes[1100];

// Array cd stores the
// starting end of pipe
int starting_vertex_of_pipes[1100];

// Vector a, b, c are used
// to store the final output
vector<int> a;
vector<int> b;
vector<int> c;

int ans;

int dfs(int w)
{
    if (starting_vertex_of_pipes[w] == 0)
        return w;
    if (diameter_between_two_pipes[w] < ans)
        ans = diameter_between_two_pipes[w];
    return dfs(starting_vertex_of_pipes[w]);
}

// Function performing calculations.
void solve(int arr[][3])
{
    for (int i = 0; i < number_of_pipes; ++i)
    {

        int house_1 = arr[i][0], house_2 = arr[i][1],
            pipe_diameter = arr[i][2];

        starting_vertex_of_pipes[house_1] = house_2;
        diameter_between_two_pipes[house_1] = pipe_diameter;
        ending_vertex_of_pipes[house_2] = house_1;
    }

    a.clear();
    b.clear();
    c.clear();

    for (int j = 1; j <= number_of_houses; ++j)

        /*If a pipe has no ending vertex
        but has starting vertex i.e is
        an outgoing pipe then we need
        to start DFS with this vertex.*/
        if (ending_vertex_of_pipes[j] == 0 && starting_vertex_of_pipes[j])
        {
            ans = 1000000000;
            int w = dfs(j);

            // We put the details of component
            // in final output array
            a.push_back(j);
            b.push_back(w);
            c.push_back(ans);
        }

    cout << a.size() << endl;
    for (int j = 0; j < a.size(); ++j)
        cout << a[j] << " " << b[j] << " " << c[j] << endl;
}
/*
5. Minimum Swaps for Bracket Balancing
You are given a string of 2N characters consisting of N ‘[‘ brackets and N ‘]’ brackets. A string is considered balanced if it can be represented in the form S2[S1] where S1 and S2 are balanced strings. We can make an unbalanced string balanced by swapping adjacent characters. Calculate the minimum number of swaps necessary to make a string balanced.
TC : O(N^2)
SC :O(1)
*/
int swapCountBruteforce(string s)
{
    // To store answer
    int ans = 0;

    // To store count of '['
    int count = 0;

    // Size of string
    int n = s.size();

    // Traverse over the string
    for (int i = 0; i < n; i++)
    {
        // When '[' encounters
        if (s[i] == '[')
        {
            count++;
        }
        // when ']' encounters
        else
        {
            count--;
        }
        // When count becomes less than 0
        if (count < 0)
        {
            // Start searching for '[' from (i+1)th index
            int j = i + 1;
            while (j < n)
            {
                // When jth index contains '['
                if (s[j] == '[')
                {
                    break;
                }
                j++;
            }
            // Increment answer
            ans += j - i;

            // Set Count to 1 again
            count = 1;

            // Bring character at jth position to ith position
            // and shift all character from i to j-1
            // towards right
            char ch = s[j];
            for (int k = j; k > i; k--)
            {
                s[k] = s[k - 1];
            }
            s[i] = ch;
        }
    }
    return ans;
}
// TC : O(N)
// SC : O(N)
long swapCountOptimal(string s)
{
    // Keep track of '['
    vector<int> pos;
    for (int i = 0; i < s.length(); ++i)
        if (s[i] == '[')
            pos.push_back(i);

    int count = 0; // To count number of encountered '['
    int p = 0;     // To track position of next '[' in pos
    long sum = 0;  // To store result

    for (int i = 0; i < s.length(); ++i)
    {
        // Increment count and move p to next position
        if (s[i] == '[')
        {
            ++count;
            ++p;
        }
        else if (s[i] == ']')
            --count;

        // We have encountered an unbalanced part of string
        if (count < 0)
        {
            // Increment sum by number of swaps required
            // i.e. position of next '[' - current position
            sum += pos[p] - i;
            swap(s[i], s[pos[p]]);
            ++p;

            // Reset count to 1
            count = 1;
        }
    }
    return sum;
}
// TC :O(N)
// SC :O(1)
long swapCountMostOptimal(string chars)
{

    // Stores total number of Left and
    // Right brackets encountered
    int countLeft = 0, countRight = 0;

    // swap stores the number of swaps
    // required imbalance maintains
    // the number of imbalance pair
    int swap = 0, imbalance = 0;

    for (int i = 0; i < chars.length(); i++)
    {
        if (chars[i] == '[')
        {

            // Increment count of Left bracket
            countLeft++;

            if (imbalance > 0)
            {

                // swaps count is last swap count + total
                // number imbalanced brackets
                swap += imbalance;

                // imbalance decremented by 1 as it solved
                // only one imbalance of Left and Right
                imbalance--;
            }
        }
        else if (chars[i] == ']')
        {

            // Increment count of Right bracket
            countRight++;

            // imbalance is reset to current difference
            // between Left and Right brackets
            imbalance = (countRight - countLeft);
        }
    }
    return swap;
}

/*
6. Egyptian Fraction
Every positive fraction can be represented as sum of unique unit fractions. A fraction is unit fraction if numerator is 1 and denominator is a positive integer, for example 1/3 is a unit fraction. Such a representation is called Egyptian Fraction as it was used by ancient Egyptians.
TC : O(d)
SC :O(1)

*/
void egyptianFractionBruteforce(int n, int d)
{
    // When Both Numerator and denominator becomes zero then we simply return;
    if (d == 0 || n == 0)
        return;
    if (d % n == 0)
    {
        cout << "1/" << d / n;
        return;
    }
    if (n % d == 0)
    {
        cout << n / d;
        return;
    }
    if (n > d)
    {
        cout << n / d << " + ";
        egyptianFractionBruteforce(n % d, d);
        return;
    }
    int x = d / n + 1;
    cout << "1/" << x << " + ";
    egyptianFractionBruteforce(n * x - d, d * x);
}
// TC :O(N^2)
// SC :O(N)
vector<int> getEgyptianFractionUtil(int numerator, int denominator,
                                    vector<int> listOfDenoms)
{
    if (numerator == 0)
        return listOfDenoms;

    int newDenom = ceil((double)denominator / numerator);

    // append in output list
    listOfDenoms.push_back(newDenom);

    listOfDenoms = getEgyptianFractionUtil(
        numerator * newDenom - denominator,
        newDenom * denominator, listOfDenoms);

    return listOfDenoms;
}
string getEgyptianFractionOptimal(int numerator, int denominator)
{
    string str = "";
    vector<int> output = getEgyptianFractionUtil(numerator, denominator, {});
    for (auto denom : output)
        str += "1/" + to_string(denom) + " + ";

    string strCopy = str.substr(0, str.length() - 3); // removing the last + sign
    return strCopy;
}

/*
7.Policemen catch thieves
Given an array of size n that has the following specifications:

Each element in the array contains either a policeman or a thief.
Each policeman can catch only one thief.
A policeman cannot catch a thief who is more than K units away from the policeman.
We need to find the maximum number of thieves that can be caught.
Input : arr[] = {'P', 'T', 'T', 'P', 'T'},
            k = 1.
Output : 2.
TC : O(N)
SC : O(N)
*/
int policeThiefBruteforce(char arr[], int n, int k)
{
    int res = 0;
    vector<int> thi;
    vector<int> pol;

    // store indices in the vector
    for (int i = 0; i < n; i++)
    {
        if (arr[i] == 'P')
            pol.push_back(i);
        else if (arr[i] == 'T')
            thi.push_back(i);
    }

    // track lowest current indices of
    // thief: thi[l], police: pol[r]
    int l = 0, r = 0;
    while (l < thi.size() && r < pol.size())
    {
        // can be caught
        if (abs(thi[l] - pol[r]) <= k)
        {
            l++;
            r++;
            res++;
        }
        // increment the minimum index
        else if (thi[l] < pol[r])
        {
            l++;
        }
        else
        {
            r++;
        }
    }
    return res;
}

// TC : O(N)
// SC : O(1)
int policeThiefOptimal(char arr[], int n, int k)
{
    // Initialize the current lowest indices of
    // policeman in pol and thief in thi variable as -1
    int pol = -1, thi = -1, res = 0;
    // Find the lowest index of policemen
    for (int i = 0; i < n; i++)
    {
        if (arr[i] == 'P')
        {
            pol = i;
            break;
        }
    }

    // Find the lowest index of thief
    for (int i = 0; i < n; i++)
    {
        if (arr[i] == 'T')
        {
            thi = i;
            break;
        }
    }

    // If lowest index of either policemen or thief remain
    // -1 then return 0
    if (thi == -1 || pol == -1)
        return 0;
    while (pol < n && thi < n)
    {
        // can be caught
        if (abs(pol - thi) <= k)
        {

            pol = pol + 1;
            while (pol < n && arr[pol] != 'P')
                pol = pol + 1;

            thi = thi + 1;
            while (thi < n && arr[thi] != 'T')
                thi = thi + 1;

            res++;
        }
        // increment the current min(pol , thi) to
        // the next policeman or thief found
        else if (thi < pol)
        {
            thi = thi + 1;
            while (thi < n && arr[thi] != 'T')
                thi = thi + 1;
        }
        else
        {
            pol = pol + 1;
            while (pol < n && arr[pol] != 'P')
                pol = pol + 1;
        }
    }
    return res;
}

/*
8. Fitting Shelves Problem
Given length of wall w and shelves of two lengths m and n, find the number of each type of shelf to be used and the remaining empty space in the optimal solution so that the empty space is minimum. The larger of the two shelves is cheaper so it is preferred. However cost is secondary and first priority is to minimize empty space on wall.
Input : w = 24 m = 3 n = 5
Output : 3 3 0
TC : O(w/max(n,m))
SC : O(1)
*/
void minSpacePreferLarge(int wall, int m, int n)
{
    // for simplicity, Assuming m is always smaller than n
    // initializing output variables
    int num_m = 0, num_n = 0, min_empty = wall;

    // p and q are no of shelves of length m and n
    // rem is the empty space
    int p = wall / m, q = 0, rem = wall % m;
    num_m = p;
    num_n = q;
    min_empty = rem;
    while (wall >= n)
    {
        // place one more shelf of length n
        q += 1;
        wall = wall - n;
        // place as many shelves of length m
        // in the remaining part
        p = wall / m;
        rem = wall % m;

        // update output variablse if curr
        // min_empty <= overall empty
        if (rem <= min_empty)
        {
            num_m = p;
            num_n = q;
            min_empty = rem;
        }
    }

    cout << num_m << " " << num_n << " "
         << min_empty << endl;
}

/*
9. Assign Mice to Holes
There are N Mice and N holes are placed in a straight line. Each hole can accommodate only 1 mouse. A mouse can stay at his position, move one step right from x to x + 1, or move one step left from x to x -1. Any of these moves consumes 1 minute. Assign mice to holes so that the time when the last mouse gets inside a hole is minimized.
TC :O(nlog(n))
SC O(1)
*/
int assignHole(int mices[], int holes[],
               int n, int m)
{

    // Base Condition
    // No. of mouse and holes should be same
    if (n != m)
        return -1;

    // Sort the arrays
    sort(mices, mices + n);
    sort(holes, holes + m);

    // Finding max difference between
    // ith mice and hole
    int max = 0;
    for (int i = 0; i < n; ++i)
    {
        if (max < abs(mices[i] - holes[i]))
            max = abs(mices[i] - holes[i]);
    }
    return max;
}
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=->>

// ================================MAIN START=================================>>
int main()
{
    // #ifndef ONLINE_JUDGE
    //     freopen("./i.txt", "r", stdin);
    //     freopen("./o.txt", "w", stdout);
    // #endif
    /*
        Some short function
           int maxi = *max_element(arr.begin(), arr.end());
            int sum = accumulate(arr.begin(), arr.end(), 0);
    */
    // Given values for 854th and 855th numbers
    // Generate all sequences from 00000 to 99999
    // Generate all sequences from 00000 to 99999
    // Generate all sequences from 00000 to 99999
    // vector<int> grid = {1, 2, 3, 4, 5};

    // vector<int> s = {1, 2, 3};
    // cout << "Total child satisfied " << findContentChildren(grid, s);
    // int n = 3, weight = 50;
    // Item arr[3] = {{100, 20}, {60, 10}, {120, 30}};
    // Solution obj;
    // double ans = obj.fractionalKnapsack(weight, arr, n);
    // cout << "The maximum value is " << setprecision(2) << fixed << ans;
    // int V = 49;
    // vector<int> ans;
    // int coins[] = {1, 2, 5, 10, 20, 50, 100, 500, 1000};
    // int n = 9;
    // cout<<"Total coin needed "<<minCoins(coins,n,V);
    // vector<int> arr = {5, 5, 5, 10, 20};
    // cout << (lemonadeChange(arr) ? "Yes" : "No") << endl;
    // cout << checkValidString("(*)");
    // vector<int> start = {4, 1, 2, 3};
    // vector<int> end = {6, 2, 4, 5};
    // Nmeet obj;
    // cout << obj.maximumMeetings(start, end);
    // vector<int> arr = {2, 3, 1, 1, 4};
    // cout << (canJump(arr) ? "Yes" : "No");
    // cout << jumpRecr(arr) << endl;
    // cout << jumpMemo(arr) << endl;
    // cout << jump(arr) << endl;
    // int arr[] = {900, 945, 955, 1100, 1500, 1800};
    // int dep[] = {920, 1200, 1130, 1150, 1900, 2000};
    // int n = sizeof(dep) / sizeof(dep[0]);
    // cout << "Minimum number of Platforms required " << countPlatformsBruteforce(n, arr, dep) << endl;
    // cout << "Minimum number of Platforms required " << countPlatformsOptimal(n, arr, dep) << endl;
    // J obj;
    // Job arr[4] = {{1, 4, 20}, {2, 1, 10}, {3, 2, 40}, {4, 2, 30}};
    // Job arr[5] = {{1, 2, 100}, {2, 1, 19}, {3, 2, 27}, {4, 1, 25}, {5, 1, 15}};

    // Convert to vector of vectors
    // vector<vector<int>> jobsVector;
    // for (int i = 0; i < 5; ++i)
    // {
    //     vector<int> job;
    //     job.push_back(arr[i].id);
    //     job.push_back(arr[i].dead);
    //     job.push_back(arr[i].profit);
    //     jobsVector.push_back(job);
    // }
    // pair<int, int> ans = JobScheduling(arr, 4);
    // cout << ans.first << " " << ans.second;
    // vector<int> ans = jobScheduling(jobsVector);
    // printVector(ans);
    // vector<int> c = {1, 0, 2};
    // cout << candy(c);
    // vector<int> a = {0, 0, 0};
    // vector<int> b = {3, 1, 2};
    // cout << sjfBetter(3, a, b) << endl;
    // cout << sjf(3, a, b) << endl;
    // VI a={3,2,4,1,7};
    // cout<<solveSjf(a);
    // int pages[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2};
    // int n = sizeof(pages) / sizeof(pages[0]);
    // int capacity = 4;
    // cout << pageFaults(pages, n, capacity);
    // vector<vector<int>> intervals;
    // intervals.push_back({1, 3});
    // intervals.push_back({6, 9});
    // vector<int> newInterval;
    // newInterval.push_back(2);
    // newInterval.push_back(5);
    // vector<vector<int>> ans = insert(intervals, newInterval);
    // vector<vector<int>> arr = {{1, 3}, {8, 10}, {2, 6}, {15, 18}};
    // vector<vector<int>> ans = mergeOverlappingIntervalsBruteforce(arr);
    // printVectorVector(ans);
    // vector<vector<int>> ans2 = mergeOverlappingIntervalsOptimal(arr);
    // printVectorVector(ans2);
    // cout << eraseOverlapIntervals(arr);
    // ===============================SOME IMPORTANT ALGORITHMS===============================>>
    // int s[] = {1, 3, 0, 5, 8, 5};
    // int f[] = {2, 4, 6, 7, 9, 9};
    // int n = sizeof(s) / sizeof(s[0]);

    // Function call
    // printMaxActivities(s, f, n);
    // number_of_houses = 9, number_of_pipes = 6;

    // memset(ending_vertex_of_pipes, 0,
    //        sizeof(ending_vertex_of_pipes));
    // memset(starting_vertex_of_pipes, 0,
    //        sizeof(starting_vertex_of_pipes));
    // memset(diameter_between_two_pipes, 0,
    //        sizeof(diameter_between_two_pipes));

    // int arr[][3] = {{7, 4, 98}, {5, 9, 72}, {4, 6, 10}, {2, 8, 22}, {9, 7, 17}, {3, 1, 66}};

    // solve(arr);
    // string s = "[]][][";
    // cout << swapCountMostOptimal(s) << "\n";
    // int numerator = 6, denominator = 14;
    // cout << "Egyptian Fraction representation of "
    //      << numerator << "/" << denominator << " is"
    //      << endl;
    // egyptianFractionBruteforce(numerator, denominator);
    // cout<<endl;
    // cout<<getEgyptianFractionOptimal(numerator,denominator);
    // char arr2[] = {'T', 'T', 'P', 'P', 'T', 'P'};
    // int k = 2;
    // int n = sizeof(arr2) / sizeof(arr2[0]);
    // cout << "Maximum thieves caught: "
    //      << policeThiefOptimal(arr2, n, k) << endl;
    // int wall = 29, m = 3, n = 9;
    // minSpacePreferLarge(wall, m, n);
    // Position of mouses
    // int mices[] = {4, -4, 2};

    // Position of holes
    // int holes[] = {4, 0, 5};

    // Number of mouses
    // int n = sizeof(mices) / sizeof(mices[0]);

    // Number of holes
    // int m = sizeof(holes) / sizeof(holes[0]);

    // The required answer is returned
    // from the function
    // int minTime = assignHole(mices, holes, n, m);

    // cout << "The last mouse gets into the hole in time:"
    //  << minTime << endl;
    // vector<int>nums={3,2,4};
    // int target=6;
    // int n = nums.size();
    // vector<int> temp = nums;
    // sort(temp.begin(), temp.end());
    // printVector(temp);
    // int left = 0, right = n;
    // while (left < right)
    // {
    //     int sum = temp[left] + temp[right];
    //     if (sum == target)
    //     {
    //       cout<<left<<" "<<right;
    //     }
    //     else if (sum < target)
    //     {
    //         left++;
    //     }
    //     else
    //         right--;
    // }

    // End code here-------->>

    return 0;
}
