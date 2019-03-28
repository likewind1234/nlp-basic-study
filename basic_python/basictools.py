global switch
switch = 0
print("====================================================")


def pause(isOneByOne = False):
    '''
    控制运行模式，分割程序段。
    :param isOneByOne:True：强行分段
    :return:
    '''
    global switch
    if (isOneByOne == False and switch >= 1):
        print("-----------------------------------------------------")
        return

    switch = switch + 1
    print("====================================================")
    input("Press Enter key to continue...")


# if __name__ == "__main__":
#     pause()