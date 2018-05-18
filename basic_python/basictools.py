global switch
switch = 0
print("====================================================")

def pause():
    global switch
    if switch >= 1:
        print("-----------------------------------------------------")
        return

    switch = switch + 1
    print("====================================================")
    input("Press Enter key to continue...")


# if __name__ == "__main__":
#     pause()