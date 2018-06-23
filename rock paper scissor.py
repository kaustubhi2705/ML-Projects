
print("...rock")
print("...paper")
print("...scissor")

player = input('Enter your choice:').lower()
print("NO CHEATING..........\n\n"*20)
import random
rand_num = random.randint(0,2)
if rand_num == 0:
    computer = 'rock'
elif rand_num == 1:
    computer = 'paper'
else:
    computer = 'scissor'
    
#print('computer plays {}'.format(computer))  # for python 3.6.4
print(f"computer plays {computer}")

if player:
    if player == computer:
        print("It's a tie")
    elif player == "rock":
        if computer == "scissor":
            print("You wins")
        elif computer == "paper":
            print("Computer wins!!")
    elif player =="paper":
        if computer == "rock":
            print("You wins")
        elif computer == "scissor":
            print("Computer wins!!")
    elif player == "scissor":
        if computer == "paper":
            print("You wins")
        elif computer == "rock":
            print("Computer wins!!")
    else:
        print("OOPS Something went wrong!!")

        