import openai
from dotenv import load_dotenv
from datasets import load_dataset
from tenacity import retry, wait_exponential, stop_after_attempt
from multiprocessing.pool import ThreadPool
from tqdm.auto import tqdm
from huggingface_hub import HfApi
import json

load_dotenv()

api = HfApi()

prompt_template = """Blocksworld is a scenario in which agent needs to stack blocks in a certain way.
The agent has four possible actions: pick up a block, put down a block, stack a block on top of another block, and unstack a block from another block.
The agent can only pick up one block at a time, and can only stack or put down a block if it is holding one.

You will be given a reasoning trace from the blocksworld problem solution. 
Each line would be numbered. You have to find the line where the agent stops examining the goal and initial state and starts thinking about the actions.

The actions can be written not using their correct names, for example, "I need to move A off of D".
You need to find such first line where the agent starts considering the actions.

Follow the format in the examples below. Answer in the correct JSON format.

Reasoning trace:
0> Okay, so I have this problem where I need to stack blocks in a specific way. Let me try to figure out the steps I need to take. 

<1> First, let me understand the initial setup. The blocks are arranged as follows: Block B is on top of Block C, which is on top of Block D. Block A is on the table, and Block D is also on the table. So, the stacks are B-C-D and A. Also, both A and B are clear, meaning nothing is on top of them, and the hand is empty.

<2> My goal is to have Block A on top of Block C, which is on top of Block D, and Block D on top of Block B. So the final stack should be A-C-D-B. Wait, no, because D is on top of B, so it's B-D-C-A? Wait, no, the goal is A on top of C, C on top of D, and D on top of B. So the stack would be A-C-D-B, with B at the bottom. But wait, D is on top of B, so the stack is B-D-C-A. Hmm, that makes sense.

<3> So, starting from B-C-D and A on the table, I need to move things around to get A on top of C, which is on top of D, which is on top of B. 

<4> Let me think about the steps. Since I can only move one block at a time and can only pick up or unstack if the block is clear. 

<5> First, I need to get Block A on top of C. But C is currently under B. So I need to move B somewhere else. Since B is on top of C, I can unstack B from C. But to do that, I need to pick up B, but B is clear, so I can unstack it. 

<6> Wait, but to unstack B, I need to have the hand empty. The hand is empty initially, so that's good. So first step: unstack B from C. Then, I can put B down somewhere. But where? Since A is on the table, maybe I can stack B on A? But wait, the goal is to have D on top of B, so maybe I should stack B on D? Wait, D is on the table, so I can stack B on D. But let me see.

<7> Wait, the initial setup is B on C, C on D, D on table, and A on table. So the stacks are B-C-D and A. So D is on the table, C is on D, B is on C. A is separate.

<8> So, to get D on top of B, I need to move D. But D is under C, which is under B. So I need to unstack C from D, but C is under B, so I can't unstack C unless I first unstack B. So first, unstack B from C, put B down on the table. Then, unstack C from D, put C down. Then, unstack D from the table? Wait, D is on the table, so I can pick up D, but I can't stack it on anything yet because I need to move it on top of B.

<9> Wait, let me outline the steps:

<10> 1. Unstack B from C. Now, B is in hand, C is on D, D is on table, A is on table.
2. Put down B on the table. Now, B is on table, C is on D, D on table, A on table.
3. Now, I need to get C on top of D, but C is already on D. Wait, no, the goal is to have A on top of C, which is on top of D, which is on top of B. So D needs to be on top of B, and C on top of D, and A on top of C.

<11> So, after step 2, B is on the table, C is on D, D is on table, A is on table.

<12> Now, I need to move D on top of B. So I can pick up D, but D is under C. So I need to unstack C from D first. But C is on D, so I can unstack C. But to do that, I need to have the hand empty. So:

<13> 3. Unstack C from D. Now, C is in hand, D is on table, B is on table, A is on table.
4. Put down C on the table. Now, C is on table, D on table, B on table, A on table.

<14> Now, I can pick up D and stack it on B. So:

<15> 5. Pick up D.
6. Stack D on top of B. Now, D is on B, B is on table, C on table, A on table.

Output:
{{"line": 5}}

Reasoning trace:
<0> Okay, so I have this block stacking problem to solve. Let me try to figure out the steps needed to get from the initial state to the goal state. I'll take it step by step because I'm still getting the hang of these kinds of problems.

<1> First, let me understand the initial conditions. The blocks involved are A, B, C, D, E, and F. The initial state is:

<2> - Block A is clear.
- Block E is clear.
- Block F is clear.
- Hand is empty.
- Block B is on top of Block D.
- Block E is on top of Block B.
- Block F is on top of Block C.
- Block A is on the table.
- Block C is on the table.
- Block D is on the table.

<3> So, visualizing this, I can imagine the stacks as:

<4> - Table has D, which has B on top, and B has E on top.
- Table has C, which has F on top.
- Table also has A, which is clear.

<5> The goal is to have:

<6> - Block A on top of Block C.
- Block B on top of Block D.
- Block C on top of Block B.
- Block E on top of Block A.
- Block F on top of Block E.

<7> Wait, that seems a bit confusing. Let me parse the goal again:

<8> - A is on top of C.
- B is on top of D.
- C is on top of B.
- E is on top of A.
- F is on top of E.

<9> So, putting that together, the final stack should be F on E, E on A, A on C, C on B, and B on D. So the order from bottom to top would be D, B, C, A, E, F.

<10> But wait, the goal says Block C is on top of Block B, which is on top of D. So D has B, which has C. Then C has A, which has E, which has F. So the stack would be D -> B -> C -> A -> E -> F.

<11> But initially, D has B and E on top, and C has F on top. A is separate on the table.

<12> So, the initial stacks are:

<13> - D: B, E
- C: F
- A: (clear)

<14> And the goal is to have:

<15> - D: B
- B: C
- C: A
- A: E
- E: F

<16> Wait, no, the goal says Block C is on top of Block B, so B is under C. So the stack would be D -> B -> C -> A -> E -> F.

<17> But initially, E is on top of B, so I need to move E off B to somewhere else. Similarly, F is on top of C, so I need to move F off C.

<18> So, the plan would involve moving E and F to the table or somewhere else temporarily so that I can rearrange the other blocks.

<19> Let me outline the steps I think are needed:

<20> 1. I need to move E off B. Since E is on top of B, which is on D, I can unstack E from B. But to do that, I need to pick up E, but E is clear, so that's possible.

<21> 2. After moving E, I can then move B off D, but wait, B is under E, so I need to move E first. Alternatively, maybe I can move E to the table.

<22> Wait, but E is on top of B, which is on D. So to unstack E, I need to have E clear, which it is, so I can unstack E from B, then put it down somewhere. But where? The table is available, so I can put E on the table.

<23> Similarly, F is on top of C, which is on the table. So I can unstack F from C, put it down on the table.

<24> Once E and F are on the table, I can then work on moving B and C.

<25> Wait, but the goal requires that C is on top of B, which is on top of D. So I need to move C on top of B, which is on D.

<26> But currently, C is on the table with F on top. So I need to move F off C first, which I can do by unstacking F and putting it down.

Output:
{{"line": 17}}

Reasoning trace:
<0> Okay, so I have this block-stacking problem to solve. Let me try to figure out the steps needed to get from the initial state to the goal state. I'll take it step by step because I'm still getting the hang of these kinds of problems.

<1> First, let me understand the initial conditions. The blocks are A, B, C, D, E, F. The hand is empty. The clear blocks are A, C, and E. That means these blocks have nothing on top of them and are on the table or on another block that's clear.

<2> Looking at the initial setup:

<3> - Block B is on top of F.
- Block C is on top of D.
- Block D is on top of B.
- So, the stack from bottom to top is F, B, D, C.
- Blocks A and E are on the table, clear.
- Block F is also on the table.

<4> So, the initial stacks are:
- F (on table) → B → D → C
- A (on table)
- E (on table)

<5> The goal is to have:
- A on top of E
- B on top of D
- C on top of B
- E on top of C
- F on top of A

<6> Wait, that seems a bit confusing. Let me parse the goal again:

<7> Goal:
- A is on top of E
- B is on top of D
- C is on top of B
- E is on top of C
- F is on top of A

<8> So, the desired stacks would be:

<9> Starting from the bottom:
- E → C → B → D
- A → F
But wait, A is on top of E, and E is on top of C. So, E is under A, but E is also on top of C. That would mean the stack is C → E → A, but then F is on top of A. So, C → E → A → F.

<10> But also, B is on top of D, and C is on top of B. So, D → B → C. But C is also part of the other stack. Hmm, that seems conflicting because C can't be in two places at once.

<11> Wait, maybe I'm misunderstanding. Let me re-express the goal:

<12> - A is on E
- B is on D
- C is on B
- E is on C
- F is on A

<13> So, E is on C, which is on B, which is on D. So, D → B → C → E → A → F.

<14> But also, A is on E, which is on C, which is on B, which is on D. So, the main stack would be D → B → C → E → A → F.

<15> But wait, that would mean F is on top of A, which is on E, which is on C, which is on B, which is on D. So, the entire stack is D at the bottom, then B, then C, then E, then A, then F on top.

<16> But the initial state has F on the table with B on top of it, and D on top of B, and C on top of D. So, the initial stack is F → B → D → C.

<17> And A and E are on the table.

<18> So, the goal is to rearrange these blocks into a single stack: D → B → C → E → A → F.

<19> Wait, but the goal also says F is on top of A, which is on E, which is on C, which is on B, which is on D. So, yes, that's the same as D → B → C → E → A → F.

<20> But how do I get there from the initial state?

<21> Let me list the initial state again:

<22> - F is on the table, with B on top, D on top of B, and C on top of D. So, stack F-B-D-C.

<23> - A and E are on the table, clear.

<24> Hand is empty.

<25> So, the goal is to have:

<26> - D on the table? Wait, no. Because in the goal, D is under B, which is under C, which is under E, which is under A, which is under F.

<27> Wait, no, the goal says F is on top of A, which is on E, which is on C, which is on B, which is on D. So, D is the base, then B, then C, then E, then A, then F.

<28> So, the stack would be D → B → C → E → A → F.

<29> But initially, D is on top of B, which is on F. So, to get D to be under B, I need to move B and D.

<30> Wait, but in the initial state, B is on F, and D is on B. So, to get B on D, I need to move D somewhere else first.

<31> This is getting a bit complicated. Let me try to break it down.

<32> First, I need to move blocks around to build the desired stack. Let's see what needs to be done step by step.

<33> 1. The goal requires that F is on top of A, which is on E, which is on C, which is on B, which is on D. So, the order from bottom to top is D, B, C, E, A, F.

<34> 2. Currently, D is on top of B, which is on F. So, to get B on top of D, I need to move D off of B. But D is on top of B, so I can unstack D from B.

<35> But wait, D is on top of B, which is on F. So, to unstack D, I need to pick up D, but D is on top of B, which is on F. So, I can unstack D from B, but I need to make sure that D is clear. Since D is on top of B, and nothing is on top of D except C, which is on top of D. So, C is on D, which is on B, which is on F.

<36> So, to unstack D, I need to first unstack C from D. Because C is on top of D, so D is not clear. So, I need to move C first.

<37> But C is clear initially, right? Wait, the initial conditions say Block C is clear. Wait, that can't be because C is on top of D, which is on top of B, which is on F. So, if C is clear, that means nothing is on top of C. So, C is on top of D, and nothing is on top of C. So, C is clear.

<38> So, I can unstack C from D because C is clear. Let me try that.

<39> Step 1: Unstack C from D. Now, C is in hand, D is now clear.

Output:
{{"line": 29}}

Reasoning trace:
<0> Okay, so I have this block stacking problem to solve. Let me try to figure out the steps needed to get from the initial state to the goal state. I'll start by understanding the initial conditions and the goal.

<1> First, the initial state:

<2> - Block A is clear.
- Hand is empty.
- Block A is on top of Block F.
- Block C is on top of Block E.
- Block D is on top of Block C.
- Block E is on top of Block B.
- Block F is on top of Block D.
- Block B is on the table.

<3> So, let me visualize this. Starting from the table, Block B is there. On top of B is E, then C, then D, then F, and finally A on top of F. So the stack from the table up is B -> E -> C -> D -> F -> A. And A is clear, meaning nothing is on top of it.

<4> The goal is:

<5> - Block A is on top of Block B.
- Block B is on top of Block C.
- Block D is on top of Block A.
- Block E is on top of Block D.
- Block F is on top of E.

<6> Wait, that seems a bit confusing. Let me parse that again.

<7> The goal states:

<8> - A is on top of B.
- B is on top of C.
- D is on top of A.
- E is on top of D.
- F is on top of E.

<9> So, the desired stack would be B -> C -> A -> D -> E -> F. But wait, that can't be right because B is on the table, so the stack would start with B, then C on B, then A on C, then D on A, E on D, and F on E. But the initial stack is B -> E -> C -> D -> F -> A. So I need to rearrange this.

<10> Wait, no, the goal is:

<11> - A is on top of B.
- B is on top of C.
- D is on top of A.
- E is on top of D.
- F is on top of E.

<12> So, the stack would be B on C, A on B, D on A, E on D, F on E. So the order from bottom to top is C -> B -> A -> D -> E -> F.

<13> But in the initial state, the stack is B -> E -> C -> D -> F -> A. So I need to move blocks around to get from the initial stack to the desired stack.

<14> Let me list the initial stack:

<15> - Table: B
  - E
    - C
      - D
        - F
          - A

<16> And the goal is:

<17> - Table: C
  - B
    - A
      - D
        - E
          - F

<18> Wait, no. The goal says:

<19> - A is on top of B.
- B is on top of C.
- D is on top of A.
- E is on top of D.
- F is on top of E.

<20> So, the stack would be C on the table, then B on C, then A on B, then D on A, then E on D, then F on E. So the order is C -> B -> A -> D -> E -> F.

<21> But in the initial state, the stack is B -> E -> C -> D -> F -> A. So I need to move blocks to get from B's stack to C's stack.

<22> Let me think about the steps needed.

<23> First, I need to get C to the table because in the goal, C is on the table. But in the initial state, C is on top of E, which is on B. So I need to unstack C from E, but to do that, I need to have E clear. But E has C on top, so I can't unstack C unless E is clear. Wait, but E is on B, which is on the table. So to unstack C from E, I need to have E clear, which it isn't because C is on it. So I need to first unstack C from E, but I can't because C is on E, and E is on B. Wait, no, the initial state says Block C is on top of Block E, which is on top of Block B. So the stack is B -> E -> C -> D -> F -> A.

<24> So, to get C to the table, I need to move E and C. But E is under C, so I need to unstack C from E, but E is on B. So perhaps I need to move E first. But E is under C, so I can't unstack E unless C is moved. So maybe I need to unstack C first.

<25> But to unstack C, I need to have C clear. Is C clear? In the initial state, Block A is clear, but what about C? C has D on top of it, so C is not clear. So I can't unstack C yet. So I need to unstack D from C first. But D is on top of C, so to unstack D, I need D to be clear. Is D clear? D has F on top, so D is not clear. So I need to unstack F from D first.

<26> Wait, F is on top of D, so to unstack F, I need F to be clear. Is F clear? F has A on top, so F is not clear. So I need to unstack A from F first.

<27> A is on top of F, and A is clear, so I can unstack A from F. Let me note that.

<28> So step 1: Unstack A from F. Now, A is in hand, F is clear.

<29> Then, I can put down A somewhere. But where? The table is occupied by B, E, C, D, F, but A is clear, so maybe I can put A on the table. But wait, the hand is empty after putting down, so I can put A on the table.

<30> Wait, but in the goal, A is on top of B, which is on top of C. So maybe I should not put A on the table yet. Alternatively, perhaps I can stack A somewhere else.

<31> Wait, but let's proceed step by step.

<32> 1. Unstack A from F. Now, A is in hand, F is clear.

<33> 2. Put down A on the table. Now, A is on the table, clear.


Output:
{{"line": 23}}

Reasoning trace:
<0> Okay, so I have this block stacking problem to solve. Let me try to figure out the steps needed to get from the initial state to the goal state. I'll take it step by step because I'm still getting the hang of these kinds of problems.

<1> First, let me understand the initial conditions. The blocks involved are A, B, C, D, E, and F. The initial state is:

<2> - Block A is clear.
- Block E is clear.
- Block F is clear.
- Hand is empty.
- Block B is on top of Block D.
- Block E is on top of Block B.
- Block F is on top of Block C.
- Block A is on the table.
- Block C is on the table.
- Block D is on the table.

<3> So, visualizing this, I can imagine the stacks as:

<4> - Table has D, which has B on top, and B has E on top.
- Table has C, which has F on top.
- Table also has A, which is clear.

<5> The goal is to have:

<6> - Block A on top of Block C.
- Block B on top of Block D.
- Block C on top of Block B.
- Block E on top of Block A.
- Block F on top of Block E.

<7> Wait, that seems a bit confusing. Let me parse the goal again:

<8> - A is on top of C.
- B is on top of D.
- C is on top of B.
- E is on top of A.
- F is on top of E.

<9> So, putting that together, the final stack should be F on E, E on A, A on C, C on B, and B on D. So the order from bottom to top would be D, B, C, A, E, F.

<10> But wait, the goal says Block C is on top of Block B, which is on top of D. So D has B, which has C. Then C has A, which has E, which has F. So the stack would be D -> B -> C -> A -> E -> F.

<11> But initially, D has B and E on top, and C has F on top. A is separate on the table.

<12> So, the initial stacks are:

<13> - D: B, E
- C: F
- A: (clear)

<14> And the goal is to have:

<15> - D: B
- B: C
- C: A
- A: E
- E: F

<16> Wait, no, the goal says Block C is on top of Block B, so B is under C. So the stack would be D -> B -> C -> A -> E -> F.

<17> But initially, E is on top of B, so I need to move E off B to somewhere else. Similarly, F is on top of C, so I need to move F off C.

<18> So, the plan would involve moving E and F to the table or somewhere else temporarily so that I can rearrange the other blocks.

<19> Let me outline the steps I think are needed:

<20> 1. I need to move E off B. Since E is on top of B, which is on D, I can unstack E from B. But to do that, I need to pick up E, but E is clear, so that's possible.

<21> 2. After moving E, I can then move B off D, but wait, B is under E, so I need to move E first. Alternatively, maybe I can move E to the table.

<22> Wait, but E is on top of B, which is on D. So to unstack E, I need to have E clear, which it is, so I can unstack E from B, then put it down somewhere. But where? The table is available, so I can put E on the table.

<23> Similarly, F is on top of C, which is on the table. So I can unstack F from C, put it down on the table.

<24> Once E and F are on the table, I can then work on moving B and C.

<25> Wait, but the goal requires that C is on top of B, which is on top of D. So I need to move C on top of B, which is on D.

<26> But currently, C is on the table with F on top. So I need to move F off C first, which I can do by unstacking F and putting it down.

<27> Similarly, E is on top of B, so I need to move E off B.

<28> So, step by step:

<29> 1. Unstack E from B, put it down on the table.

<30> 2. Unstack F from C, put it down on the table.

Output:
{{"line": 17}}

Reasoining trace:
{step}

Output:
"""

prompt_template = """Blocksworld is a scenario in which agent needs to stack blocks in a certain way.
The agent has four possible actions: pick up a block, put down a block, stack a block on top of another block, and unstack a block from another block.
The agent can only pick up one block at a time, and can only stack or put down a block if it is holding one.

You will be given a reasoning trace from the blocksworld problem solution. 
Each line would be numbered. You have to find the line where the agent stops examining the goal and initial state.

You need to find such first line where the agent stops being confused about the goal and initial state and starts thinking about the actions.

Follow the format in the examples below. Answer in the correct JSON format.

Reasoning trace:
0> Okay, so I have this problem where I need to stack blocks in a specific way. Let me try to figure out the steps I need to take. 

<1> First, let me understand the initial setup. The blocks are arranged as follows: Block B is on top of Block C, which is on top of Block D. Block A is on the table, and Block D is also on the table. So, the stacks are B-C-D and A. Also, both A and B are clear, meaning nothing is on top of them, and the hand is empty.

<2> My goal is to have Block A on top of Block C, which is on top of Block D, and Block D on top of Block B. So the final stack should be A-C-D-B. Wait, no, because D is on top of B, so it's B-D-C-A? Wait, no, the goal is A on top of C, C on top of D, and D on top of B. So the stack would be A-C-D-B, with B at the bottom. But wait, D is on top of B, so the stack is B-D-C-A. Hmm, that makes sense.

<3> So, starting from B-C-D and A on the table, I need to move things around to get A on top of C, which is on top of D, which is on top of B. 

<4> Let me think about the steps. Since I can only move one block at a time and can only pick up or unstack if the block is clear. 

<5> First, I need to get Block A on top of C. But C is currently under B. So I need to move B somewhere else. Since B is on top of C, I can unstack B from C. But to do that, I need to pick up B, but B is clear, so I can unstack it. 

<6> Wait, but to unstack B, I need to have the hand empty. The hand is empty initially, so that's good. So first step: unstack B from C. Then, I can put B down somewhere. But where? Since A is on the table, maybe I can stack B on A? But wait, the goal is to have D on top of B, so maybe I should stack B on D? Wait, D is on the table, so I can stack B on D. But let me see.

<7> Wait, the initial setup is B on C, C on D, D on table, and A on table. So the stacks are B-C-D and A. So D is on the table, C is on D, B is on C. A is separate.

<8> So, to get D on top of B, I need to move D. But D is under C, which is under B. So I need to unstack C from D, but C is under B, so I can't unstack C unless I first unstack B. So first, unstack B from C, put B down on the table. Then, unstack C from D, put C down. Then, unstack D from the table? Wait, D is on the table, so I can pick up D, but I can't stack it on anything yet because I need to move it on top of B.

<9> Wait, let me outline the steps:

<10> 1. Unstack B from C. Now, B is in hand, C is on D, D is on table, A is on table.
2. Put down B on the table. Now, B is on table, C is on D, D on table, A on table.
3. Now, I need to get C on top of D, but C is already on D. Wait, no, the goal is to have A on top of C, which is on top of D, which is on top of B. So D needs to be on top of B, and C on top of D, and A on top of C.

<11> So, after step 2, B is on the table, C is on D, D is on table, A is on table.

<12> Now, I need to move D on top of B. So I can pick up D, but D is under C. So I need to unstack C from D first. But C is on D, so I can unstack C. But to do that, I need to have the hand empty. So:

<13> 3. Unstack C from D. Now, C is in hand, D is on table, B is on table, A is on table.
4. Put down C on the table. Now, C is on table, D on table, B on table, A on table.

<14> Now, I can pick up D and stack it on B. So:

<15> 5. Pick up D.
6. Stack D on top of B. Now, D is on B, B is on table, C on table, A on table.

Output:
{{"line": 5}}

Reasoning trace:
<0> Okay, so I have this block stacking problem to solve. Let me try to figure out the steps needed to get from the initial state to the goal state. I'll take it step by step because I'm still getting the hang of these kinds of problems.

<1> First, let me understand the initial conditions. The blocks involved are A, B, C, D, E, and F. The initial state is:

<2> - Block A is clear.
- Block E is clear.
- Block F is clear.
- Hand is empty.
- Block B is on top of Block D.
- Block E is on top of Block B.
- Block F is on top of Block C.
- Block A is on the table.
- Block C is on the table.
- Block D is on the table.

<3> So, visualizing this, I can imagine the stacks as:

<4> - Table has D, which has B on top, and B has E on top.
- Table has C, which has F on top.
- Table also has A, which is clear.

<5> The goal is to have:

<6> - Block A on top of Block C.
- Block B on top of Block D.
- Block C on top of Block B.
- Block E on top of Block A.
- Block F on top of Block E.

<7> Wait, that seems a bit confusing. Let me parse the goal again:

<8> - A is on top of C.
- B is on top of D.
- C is on top of B.
- E is on top of A.
- F is on top of E.

<9> So, putting that together, the final stack should be F on E, E on A, A on C, C on B, and B on D. So the order from bottom to top would be D, B, C, A, E, F.

<10> But wait, the goal says Block C is on top of Block B, which is on top of D. So D has B, which has C. Then C has A, which has E, which has F. So the stack would be D -> B -> C -> A -> E -> F.

<11> But initially, D has B and E on top, and C has F on top. A is separate on the table.

<12> So, the initial stacks are:

<13> - D: B, E
- C: F
- A: (clear)

<14> And the goal is to have:

<15> - D: B
- B: C
- C: A
- A: E
- E: F

<16> Wait, no, the goal says Block C is on top of Block B, so B is under C. So the stack would be D -> B -> C -> A -> E -> F.

<17> But initially, E is on top of B, so I need to move E off B to somewhere else. Similarly, F is on top of C, so I need to move F off C.

<18> So, the plan would involve moving E and F to the table or somewhere else temporarily so that I can rearrange the other blocks.

<19> Let me outline the steps I think are needed:

<20> 1. I need to move E off B. Since E is on top of B, which is on D, I can unstack E from B. But to do that, I need to pick up E, but E is clear, so that's possible.

<21> 2. After moving E, I can then move B off D, but wait, B is under E, so I need to move E first. Alternatively, maybe I can move E to the table.

<22> Wait, but E is on top of B, which is on D. So to unstack E, I need to have E clear, which it is, so I can unstack E from B, then put it down somewhere. But where? The table is available, so I can put E on the table.

<23> Similarly, F is on top of C, which is on the table. So I can unstack F from C, put it down on the table.

<24> Once E and F are on the table, I can then work on moving B and C.

<25> Wait, but the goal requires that C is on top of B, which is on top of D. So I need to move C on top of B, which is on D.

<26> But currently, C is on the table with F on top. So I need to move F off C first, which I can do by unstacking F and putting it down.

Output:
{{"line": 18}}

Reasoning trace:
<0> Okay, so I have this block-stacking problem to solve. Let me try to figure out the steps needed to get from the initial state to the goal state. I'll take it step by step because I'm still getting the hang of these kinds of problems.

<1> First, let me understand the initial conditions. The blocks are A, B, C, D, E, F. The hand is empty. The clear blocks are A, C, and E. That means these blocks have nothing on top of them and are on the table or on another block that's clear.

<2> Looking at the initial setup:

<3> - Block B is on top of F.
- Block C is on top of D.
- Block D is on top of B.
- So, the stack from bottom to top is F, B, D, C.
- Blocks A and E are on the table, clear.
- Block F is also on the table.

<4> So, the initial stacks are:
- F (on table) → B → D → C
- A (on table)
- E (on table)

<5> The goal is to have:
- A on top of E
- B on top of D
- C on top of B
- E on top of C
- F on top of A

<6> Wait, that seems a bit confusing. Let me parse the goal again:

<7> Goal:
- A is on top of E
- B is on top of D
- C is on top of B
- E is on top of C
- F is on top of A

<8> So, the desired stacks would be:

<9> Starting from the bottom:
- E → C → B → D
- A → F
But wait, A is on top of E, and E is on top of C. So, E is under A, but E is also on top of C. That would mean the stack is C → E → A, but then F is on top of A. So, C → E → A → F.

<10> But also, B is on top of D, and C is on top of B. So, D → B → C. But C is also part of the other stack. Hmm, that seems conflicting because C can't be in two places at once.

<11> Wait, maybe I'm misunderstanding. Let me re-express the goal:

<12> - A is on E
- B is on D
- C is on B
- E is on C
- F is on A

<13> So, E is on C, which is on B, which is on D. So, D → B → C → E → A → F.

<14> But also, A is on E, which is on C, which is on B, which is on D. So, the main stack would be D → B → C → E → A → F.

<15> But wait, that would mean F is on top of A, which is on E, which is on C, which is on B, which is on D. So, the entire stack is D at the bottom, then B, then C, then E, then A, then F on top.

<16> But the initial state has F on the table with B on top of it, and D on top of B, and C on top of D. So, the initial stack is F → B → D → C.

<17> And A and E are on the table.

<18> So, the goal is to rearrange these blocks into a single stack: D → B → C → E → A → F.

<19> Wait, but the goal also says F is on top of A, which is on E, which is on C, which is on B, which is on D. So, yes, that's the same as D → B → C → E → A → F.

<20> But how do I get there from the initial state?

<21> Let me list the initial state again:

<22> - F is on the table, with B on top, D on top of B, and C on top of D. So, stack F-B-D-C.

<23> - A and E are on the table, clear.

<24> Hand is empty.

<25> So, the goal is to have:

<26> - D on the table? Wait, no. Because in the goal, D is under B, which is under C, which is under E, which is under A, which is under F.

<27> Wait, no, the goal says F is on top of A, which is on E, which is on C, which is on B, which is on D. So, D is the base, then B, then C, then E, then A, then F.

<28> So, the stack would be D → B → C → E → A → F.

<29> But initially, D is on top of B, which is on F. So, to get D to be under B, I need to move B and D.

<30> Wait, but in the initial state, B is on F, and D is on B. So, to get B on D, I need to move D somewhere else first.

<31> This is getting a bit complicated. Let me try to break it down.

<32> First, I need to move blocks around to build the desired stack. Let's see what needs to be done step by step.

<33> 1. The goal requires that F is on top of A, which is on E, which is on C, which is on B, which is on D. So, the order from bottom to top is D, B, C, E, A, F.

<34> 2. Currently, D is on top of B, which is on F. So, to get B on top of D, I need to move D off of B. But D is on top of B, so I can unstack D from B.

<35> But wait, D is on top of B, which is on F. So, to unstack D, I need to pick up D, but D is on top of B, which is on F. So, I can unstack D from B, but I need to make sure that D is clear. Since D is on top of B, and nothing is on top of D except C, which is on top of D. So, C is on D, which is on B, which is on F.

<36> So, to unstack D, I need to first unstack C from D. Because C is on top of D, so D is not clear. So, I need to move C first.

<37> But C is clear initially, right? Wait, the initial conditions say Block C is clear. Wait, that can't be because C is on top of D, which is on top of B, which is on F. So, if C is clear, that means nothing is on top of C. So, C is on top of D, and nothing is on top of C. So, C is clear.

<38> So, I can unstack C from D because C is clear. Let me try that.

<39> Step 1: Unstack C from D. Now, C is in hand, D is now clear.

Output:
{{"line": 32}}

Reasoning trace:
<0> Okay, so I have this block stacking problem to solve. Let me try to figure out the steps needed to get from the initial state to the goal state. I'll start by understanding the initial conditions and the goal.

<1> First, the initial state:

<2> - Block A is clear.
- Hand is empty.
- Block A is on top of Block F.
- Block C is on top of Block E.
- Block D is on top of Block C.
- Block E is on top of Block B.
- Block F is on top of Block D.
- Block B is on the table.

<3> So, let me visualize this. Starting from the table, Block B is there. On top of B is E, then C, then D, then F, and finally A on top of F. So the stack from the table up is B -> E -> C -> D -> F -> A. And A is clear, meaning nothing is on top of it.

<4> The goal is:

<5> - Block A is on top of Block B.
- Block B is on top of Block C.
- Block D is on top of Block A.
- Block E is on top of Block D.
- Block F is on top of E.

<6> Wait, that seems a bit confusing. Let me parse that again.

<7> The goal states:

<8> - A is on top of B.
- B is on top of C.
- D is on top of A.
- E is on top of D.
- F is on top of E.

<9> So, the desired stack would be B -> C -> A -> D -> E -> F. But wait, that can't be right because B is on the table, so the stack would start with B, then C on B, then A on C, then D on A, E on D, and F on E. But the initial stack is B -> E -> C -> D -> F -> A. So I need to rearrange this.

<10> Wait, no, the goal is:

<11> - A is on top of B.
- B is on top of C.
- D is on top of A.
- E is on top of D.
- F is on top of E.

<12> So, the stack would be B on C, A on B, D on A, E on D, F on E. So the order from bottom to top is C -> B -> A -> D -> E -> F.

<13> But in the initial state, the stack is B -> E -> C -> D -> F -> A. So I need to move blocks around to get from the initial stack to the desired stack.

<14> Let me list the initial stack:

<15> - Table: B
  - E
    - C
      - D
        - F
          - A

<16> And the goal is:

<17> - Table: C
  - B
    - A
      - D
        - E
          - F

<18> Wait, no. The goal says:

<19> - A is on top of B.
- B is on top of C.
- D is on top of A.
- E is on top of D.
- F is on top of E.

<20> So, the stack would be C on the table, then B on C, then A on B, then D on A, then E on D, then F on E. So the order is C -> B -> A -> D -> E -> F.

<21> But in the initial state, the stack is B -> E -> C -> D -> F -> A. So I need to move blocks to get from B's stack to C's stack.

<22> Let me think about the steps needed.

<23> First, I need to get C to the table because in the goal, C is on the table. But in the initial state, C is on top of E, which is on B. So I need to unstack C from E, but to do that, I need to have E clear. But E has C on top, so I can't unstack C unless E is clear. Wait, but E is on B, which is on the table. So to unstack C from E, I need to have E clear, which it isn't because C is on it. So I need to first unstack C from E, but I can't because C is on E, and E is on B. Wait, no, the initial state says Block C is on top of Block E, which is on top of Block B. So the stack is B -> E -> C -> D -> F -> A.

<24> So, to get C to the table, I need to move E and C. But E is under C, so I need to unstack C from E, but E is on B. So perhaps I need to move E first. But E is under C, so I can't unstack E unless C is moved. So maybe I need to unstack C first.

<25> But to unstack C, I need to have C clear. Is C clear? In the initial state, Block A is clear, but what about C? C has D on top of it, so C is not clear. So I can't unstack C yet. So I need to unstack D from C first. But D is on top of C, so to unstack D, I need D to be clear. Is D clear? D has F on top, so D is not clear. So I need to unstack F from D first.

<26> Wait, F is on top of D, so to unstack F, I need F to be clear. Is F clear? F has A on top, so F is not clear. So I need to unstack A from F first.

<27> A is on top of F, and A is clear, so I can unstack A from F. Let me note that.

<28> So step 1: Unstack A from F. Now, A is in hand, F is clear.

<29> Then, I can put down A somewhere. But where? The table is occupied by B, E, C, D, F, but A is clear, so maybe I can put A on the table. But wait, the hand is empty after putting down, so I can put A on the table.

<30> Wait, but in the goal, A is on top of B, which is on top of C. So maybe I should not put A on the table yet. Alternatively, perhaps I can stack A somewhere else.

<31> Wait, but let's proceed step by step.

<32> 1. Unstack A from F. Now, A is in hand, F is clear.

<33> 2. Put down A on the table. Now, A is on the table, clear.


Output:
{{"line": 22}}

Reasoning trace:
<0> Okay, so I have this block stacking problem to solve. Let me try to figure out the steps needed to get from the initial state to the goal state. I'll take it step by step because I'm still getting the hang of these kinds of problems.

<1> First, let me understand the initial conditions. The blocks involved are A, B, C, D, E, and F. The initial state is:

<2> - Block A is clear.
- Block E is clear.
- Block F is clear.
- Hand is empty.
- Block B is on top of Block D.
- Block E is on top of Block B.
- Block F is on top of Block C.
- Block A is on the table.
- Block C is on the table.
- Block D is on the table.

<3> So, visualizing this, I can imagine the stacks as:

<4> - Table has D, which has B on top, and B has E on top.
- Table has C, which has F on top.
- Table also has A, which is clear.

<5> The goal is to have:

<6> - Block A on top of Block C.
- Block B on top of Block D.
- Block C on top of Block B.
- Block E on top of Block A.
- Block F on top of Block E.

<7> Wait, that seems a bit confusing. Let me parse the goal again:

<8> - A is on top of C.
- B is on top of D.
- C is on top of B.
- E is on top of A.
- F is on top of E.

<9> So, putting that together, the final stack should be F on E, E on A, A on C, C on B, and B on D. So the order from bottom to top would be D, B, C, A, E, F.

<10> But wait, the goal says Block C is on top of Block B, which is on top of D. So D has B, which has C. Then C has A, which has E, which has F. So the stack would be D -> B -> C -> A -> E -> F.

<11> But initially, D has B and E on top, and C has F on top. A is separate on the table.

<12> So, the initial stacks are:

<13> - D: B, E
- C: F
- A: (clear)

<14> And the goal is to have:

<15> - D: B
- B: C
- C: A
- A: E
- E: F

<16> Wait, no, the goal says Block C is on top of Block B, so B is under C. So the stack would be D -> B -> C -> A -> E -> F.

<17> But initially, E is on top of B, so I need to move E off B to somewhere else. Similarly, F is on top of C, so I need to move F off C.

<18> So, the plan would involve moving E and F to the table or somewhere else temporarily so that I can rearrange the other blocks.

<19> Let me outline the steps I think are needed:

<20> 1. I need to move E off B. Since E is on top of B, which is on D, I can unstack E from B. But to do that, I need to pick up E, but E is clear, so that's possible.

<21> 2. After moving E, I can then move B off D, but wait, B is under E, so I need to move E first. Alternatively, maybe I can move E to the table.

<22> Wait, but E is on top of B, which is on D. So to unstack E, I need to have E clear, which it is, so I can unstack E from B, then put it down somewhere. But where? The table is available, so I can put E on the table.

<23> Similarly, F is on top of C, which is on the table. So I can unstack F from C, put it down on the table.

<24> Once E and F are on the table, I can then work on moving B and C.

<25> Wait, but the goal requires that C is on top of B, which is on top of D. So I need to move C on top of B, which is on D.

<26> But currently, C is on the table with F on top. So I need to move F off C first, which I can do by unstacking F and putting it down.

<27> Similarly, E is on top of B, so I need to move E off B.

<28> So, step by step:

<29> 1. Unstack E from B, put it down on the table.

<30> 2. Unstack F from C, put it down on the table.

Output:
{{"line": 19}}

Reasoining trace:
{step}

Output:
"""

client = openai.Client(
    base_url="https://openrouter.ai/api/v1",
)


from pydantic import BaseModel

class ExtractedActions(BaseModel):
    actions: list[list[str]] | None


def check_step(step):
    keywords = ["pick up", "put down", "stack", "unstack"]
    for keyword in keywords:
        if keyword in step.lower():
            return True


def thread_fn(step):
    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3))
    def gen_label(step):
        prompt = prompt_template.format(step=step)
        # if check_step(step):
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_tokens=200,
            response_format={ "type": "json_object" },
        )

        label = json.loads(response.choices[0].message.content)

        return label
    
    return gen_label(step)

def process_item(item):
    text = item["generation"]
    steps = text.split("\n\n")

    text = "\n\n".join(
        f"<{i}> {step}" for i, step in enumerate(steps[:50])
    ) 

    try:
        label = thread_fn(text)
    except Exception as e:
        print(e)
        label = None

    return {
        "index": item["index"],
        "label": label
    }

def main(start, end, n_threads, save_name):
    blocksworld_type = "6-blocks-big"
    dataset = load_dataset(f"dmitriihook/deepseek-r1-qwen-32b-planning-{blocksworld_type}")["train"]

    dataset = dataset.add_column("index", [i for i in range(len(dataset))])
    items = dataset.select(range(start, end))
    with ThreadPool(n_threads) as pool:
        results = list(tqdm(pool.imap(process_item, items), total=end - start))

    with open(f"{save_name}.json", "w") as f:
        json.dump(results, f)   


    try:
        api.create_repo(f"dmitriihook/{save_name}", repo_type="dataset")
    except Exception as e:
        print(e)
    
    api.upload_file(
        repo_id=f"dmitriihook/{save_name}",
        path_or_fileobj=f"{save_name}.json",
        path_in_repo=f"{save_name}.json",
        repo_type="dataset"
    )

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=5000)
parser.add_argument("--n_threads", type=int, default=20)
parser.add_argument("--save_name", type=str, default="blocksworld-6-blocks-actions-first-combined-v2")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.start, args.end, args.n_threads, args.save_name)

    

    
    