# DeepCNN-as-a-Service
Selection and Optimization of Deep Convolutional Neural Networks as a Service.

Yanis Tazi & Erik Weissenborn

-------------------------
Motivation
-------------------------
Model selection, training, and finetuning of a Convolutional Neural Net is a time consuming and tedious task. It often requires multiple repetitions of training with various hyperparameter configurations. We decided to formalize this process by constructing a cloud service that would enable the user to upload a previously unseen labeled dataset and return to the user a highly accurate and finetuned CNN that is ready for use. The user will be emailed a report, which displays the finetuned model’s training and validation accuracy, training loss, and a confusion matrix with test results. The cloud solution offers an efficient way for users to obtain a finetuned CNN model that they can begin using out of the box for inference.




Class Hierachy

		|-------------------------|             |------------|
                |     BinarySearchTree    |--has-a----->|   BSTNode  |
                |-------------------------|             |------------|
                           ^    ^                           ^  ^
                          /      \                          |  |
                       is-a     is-a                        |  |
                        /          \                        |  |
                       /           |--------------|         |  |
                      /            |  SplayTree   |-has-a---|  |
                     /             |--------------|            |
                    /                                        is-a 
                   /                                           |
    |----------------|                                  |------------|
    |     AVLTree    |--has-a-------------------------->| AVLNode    |
    |----------------|                                  |------------|
   



Q.  There is no SplayNode class.  Why does SplayTree use BSTNodes?

A.  The SplayTree class uses BSTNode, while the AVLTree uses AVLNode. The idea
is that AVLTree needs to keep track of the height information. On the other
hand, SplayTree doesn't need to have height information. This is very similar
to BSTNode class, so there's no need to make a SplayNode class.


Q.  What are the similarities and differences between AVLTree rotation methods
and SplayTree rotation methods?  Why are they not methods of BinarySearchTree
class or the node classes?

A.  They are basically the same. However, for the AVLTree rotations, they need
to update the height information while the SplayTree's methods don't. These
methods could have been methods of BinarySearchTree, so that AVLTree and
SplayTree can inherit them; however, BST doesn't use rotation at all. We then
came to the decision of making each of the AVLTree, SplayTree different
rotation methods.


Q.  The Heap class does not allocate any memory--it shares the data.  Why?

A.  It makes it faster =)


Q.  Sort() is currently a method of HeapSort.  The alternative is to define an
external HeapSort() function that is a friend of the Heap class--HeapSort()
still needs to call the PercDown() function.  Why did we choose to make
HeapSort a function inside the Heap class?

A.  We think that our special(ized) Heap should know how to sort itself. :)


-----------------------
Word Frequency Analysis
-----------------------

These were the steps to the process:
    1 ) run word-count on each essay
    2 ) record the frequency for the top 28 most frequent words and
	take a subtotal for each (this will be the total number of
	'relevant' words in each literature)
    3 ) isolate the words that made it to top 7 (there were 10), and
	take their percentage with respect to the subtotals
    4 ) anaylze by comparing the percentages

Of the 10 words to anaylze, words not used at predictable frequencies by Sir
Francis Bacon (is, in, you) were ignored.  That left 6 words that are used
at predictable frequencies by Bacon (of, the, and, that, to, a), and 1 word
that is not used significantly by Bacon (I).  Two of the 6 words used
frequently and predictably by Bacon (to, a) were used just as frequent
and predictable in Hamlet and All's Well that Ends Well.  On the other hand,
the 4 remaining words (of, the, and, that), which also happens to be the words
used the most frequent by Bacon, are used at only 1/2 to 2/3 the frequency by
the author of Hamlet and All's Well that Ends Well.  The minor similarities
hardly make up for the major differeces.  Based on the above analysis, we 
conclude that Bacon wrote neither of Shakespeare's plays.
	

---------
Profiling
---------

Our expected performance bottlenecks for the word-count program are...
	+ the Splay operation for SplayTree
	+ the insertHelp function for AVL, and
	+ the PercolateDown operation for Heap
We chose these to be the bottlenecks because we assumed the function call
frequencies and costs. However, it turned out that some of these are not huge
bottlenecks at all. For example, Splay funtion doesn't take as much time as we
thought.  Words that are inserted frequently don't take so much rotations
because they are already close to the root.


  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ns/call  ns/call  name    

  1.05      1.55     0.02                             SplayTree<>::Splay()
  4.26      0.45     0.10	                      AVLTree<>::insertHelp()
  1.28      1.76     0.03                             Heap::percDown()


The gprof results show that, in general standard I/O operations takes up most 
of the runtime:


index % time    self  children    called     name
[2]     24.3    0.05    0.52   45429         next_token()
[3]     20.0    0.05    0.42                 operator<<()


Even though each word is only read/printed once, we think the cause for I/O
as a bottleneck is its huge constant overhead (which much exceeds the log n
tree operations)

For specific choice of trees and input, namely bst tree with word.txt as
input, the biggest bottleneck was string operations and FindNode() function. 
The cause is that the pre-sorted words created an unbalanced binary search
tree, requiring a complete traversal of a linked-list-like data structure
everytime a word is inserted.  At each node visited during the traversal,
FindNode compares the string keys.


index % time    self  children    called     name
                                                 <spontaneous>
[1]     75.6  167.51  533.04                 std::string::compare()
               99.37  235.36 3408496094/3408541521     std::string::size()
               35.05   81.63 1704248047/1704293468     std::string::data() 
               81.63    0.00 1704248047/2523092351     std::string::_M_data()


Another significant operation which we overlooked (for the AVLTree) is the
UpdateHeight function (which actually took 3.8 percent of the run time).

  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ns/call  ns/call  name    
  3.83      0.54     0.09                             AVLTree<>::updateHeight()


--------------------
Algorithmic Analysis
--------------------

From our "SelectionSortSort vs. HeapSort" plot, we can see that when N is greater than
8000, selectionsort function grows faster than heapsort function. This confroms our knowledge
that heap sort has better performance than selecion sort in terms of time.

We know that both mergesort(f) and heapsort(g)  have O(n logn) runing time.
In our "MergeSort vs. HeapSort" plot, two lines are parallel to each other most of the time.
This shows that f/g = constant, which means that the runing time of the two algorithms are different
by a constant factor.

------------------------------
Word Stemming and Punctuations
------------------------------
 For the punctuation part, we use the built in function "ispunct(c)" to check if 
the character is a punctuation or a character. If it is a punctuation, then 
just ignore (treat it like a space). However, due to the nature of 
the ispunct(c) function - it considers " ' " and " - " as
punctuations - there are chances for the side effects happen: "I'd" will be read as 2 
different words: "I" and "d", same as "he's" becomes "he" and "s". The other case can 
be "ice-cream" becomes "ice" and "cream". Therefore, we came to the decision of 
using one of our own functions--ispunctuation(c)--to eliminate the case 's and 'd, 
and '-'. This function will call ispunct(c) if c is not the ' and -. 

 For the word-stemming part, two stemming kinds are taken care: 
words end with -s and with -ly. Due to our language skill limitations,
we cannot check all the cases and all the exceptions. We have tried our best to check 
all of the cases that we think of. Also, since " 'd ", " 's " is in the scope above,
we remove them as well. For this option, one needs to add "-suf" in the command line as
mentioned above.
