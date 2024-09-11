2. Briefly explain your approach
My initial thought is to use cosine similarity for all detections and use only that to track object. But upon consideration, this approach time complexity is too high O(n square).

I want to reduce it by just using 1 camera at a time, but it is still O(n square) even though the time complexity could be improved by 25 times averagely (5 square)

I want to reduce it more, so I want to use something that can represent an object. Average is my initial thought, I attempt to track it on the flight but found that conditions for new object can become really messy(*). So, I just K means of 5 clusters to represent the object average.

Ofcourse, K means has some random there, so seed/random state is added for reproducible solution.

Not all feature can be correctly mapped just by feature, so, position tracking is added as an extra step to map dectection and feature. I choose position because I did something similar in the past.

3. My solution requires all data to be available (because I use K means clustering). Here is how I could change my approach to tackle this issue.

- As stated in (*), I attempted to track it on the flight and average out the new detection. This will not need all of the data but there has to be a mechanism to merge multiple trackers of the same object.

- If camera installed position is known, an inter-camera mapping can be implemented. A position of a bbox from one camera can go through certain transformation to have the estimated position on others. This can be done only if there is assumption we can make from the bbox position.