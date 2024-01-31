#include <bits/stdc++.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>
#include <vector>
using namespace std;

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

Each child i has a greed factor g[i], which is the minimum size of a cookie that the child will be content with; and each cookie j has a size s[j]. If s[j] >= g[i], we can assign the cookie j to the child i, and the child i will be content. Your goal is to maximize the number of your content children and output the maximum number.
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
int findContentChildren(vector<int> &g, vector<int> &s)
{
    int m = g.size();
    int n = s.size();
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    int i = 0, j = 0;
    while (i < m && j < n)
    {
        if (g[i] <= s[j])
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
struct Item
{
    int value;
    int weight;
};
class Solution
{
public:
    bool static comp(Item a, Item b)
    {
        double r1 = (double)a.value / (double)a.weight;
        double r2 = (double)b.value / (double)b.weight;
        return r1 > r2;
    }
    // function to return fractionalweights
    double fractionalKnapsack(int W, Item arr[], int n)
    {

        sort(arr, arr + n, comp);

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
Input :   || Output :
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
bool lemonadeChange(vector<int> &bills)
{
    int count5 = 0, count10 = 0;
    for (int bill : bills)
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
// Better ----------->
// TC :
// SC :
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
            minOpen = std::max(minOpen - 1, 0);
            maxOpen--;
        }
        else
        { // '*'
            minOpen = std::max(minOpen - 1, 0);
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

        // Initialize meetings
        for (int i = 0; i < n; i++)
        {
            meet[i].start = start[i];
            meet[i].end = end[i];
            meet[i].pos = i + 1;
        }

        // Sort meetings based on end times
        sort(meet.begin(), meet.end(), comparator);

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
bool canJump(vector<int> &nums)
{
    int furthestReachable = 0;
    int n = nums.size();
    for (int i = 0; i < n; ++i)
    {
        // Update furthest reachable index based on current element
        if (i > furthestReachable)
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
// TC :
// SC :
// Better ----------->
// TC :
// SC :
// Optimal ---------->
// TC :O(N)
// SC :O(1)
int jump(vector<int> &nums)
{
    int n = nums.size();
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

        sort(arr, arr + n, comparison);
        int maxi = arr[0].dead;
        for (int i = 1; i < n; i++)
        {
            maxi = max(maxi, arr[i].dead);
        }

        int slot[maxi + 1];

        for (int i = 0; i <= maxi; i++)
            slot[i] = -1;

        int countJobs = 0, jobProfit = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = arr[i].dead; j > 0; j--)
            {
                if (slot[j] == -1)
                {
                    slot[j] = i;
                    countJobs++;
                    jobProfit += arr[i].profit;
                    break;
                }
            }
        }

        return make_pair(countJobs, jobProfit);
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
    // cout << lemonadeChange(arr);
    // cout << checkValidString("(*)");
    // vector<int> start = {1, 3, 0, 5, 8, 5};
    // vector<int> end = {2, 4, 5, 7, 9, 9};
    // Nmeet obj;
    // cout << obj.maximumMeetings(start, end);
    // vector<int> arr = {1, 2, 0, 0, 3};
    // cout << (canJump(arr) ? "Yes" : "No");
    // int arr[] = {900, 945, 955, 1100, 1500, 1800};
    // int dep[] = {920, 1200, 1130, 1150, 1900, 2000};
    // int n = sizeof(dep) / sizeof(dep[0]);
    // cout << "Minimum number of Platforms required " << countPlatformsBruteforce(n, arr, dep) << endl;
    // cout << "Minimum number of Platforms required " << countPlatformsOptimal(n, arr, dep) << endl;
    // J obj;
    // Job arr[4] = {{1, 4, 20}, {2, 1, 10}, {3, 2, 40}, {4, 2, 30}};
    // Job arr[4] = {{1, 4, 20}, {2, 1, 10}, {3, 2, 40}, {4, 2, 30}};

    // Convert to vector of vectors
    // vector<vector<int>> jobsVector;
    // for (int i = 0; i < 4; ++i)
    // {
    //     vector<int> job;
    //     job.push_back(arr[i].id);
    //     job.push_back(arr[i].dead);
    //     job.push_back(arr[i].profit);
    //     jobsVector.push_back(job);
    // }
    // pair<int, int> ans = obj.JobScheduling(arr, 4);
    // vector<int> ans = jobScheduling(jobsVector);
    // cout << ans.first << " " << ans.second;
    // printVector(ans);
    // vector<int> c = {1, 0, 2};
    // cout << candy(c);
    // vector<int> a = {0, 0, 0};
    // vector<int> b = {3, 1, 2};
    // cout << sjfBetter(3, a, b) << endl;
    // cout << sjf(3, a, b) << endl;
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
    vector<vector<int>> arr = {{1, 3}, {8, 10}, {2, 6}, {15, 18}};
    // vector<vector<int>> ans = mergeOverlappingIntervalsBruteforce(arr);
    // printVectorVector(ans);
    // vector<vector<int>> ans2 = mergeOverlappingIntervalsOptimal(arr);
    // printVectorVector(ans2);
    cout<<eraseOverlapIntervals(arr);

    // End code here-------->>

    return 0;
}
