import re
import sys

from data_manipulation import cluster
from data_manipulation import genre_list
from data_manipulation import genre_list_multiple_users
from data_manipulation import genre_movies
from data_manipulation import highest_rating
from data_manipulation import highest_views
from data_manipulation import movies_id
from data_manipulation import movies_title
from data_manipulation import movies_watched
from data_manipulation import movies_year
from data_manipulation import user_comparison
from data_manipulation import user_history
from data_manipulation import user_recommendation

exit = False
useSmallDataSet = True


def printOptions():
    print('*** Movie DB Explorer ***')
    print('OPTIONS:')
    print('1. exit to end, small to switch to the small dataset, large to switch to the large dataset')
    print('2. movies_id id')
    print('3. movies_title "title"')
    print('4. movies_year year')
    print('5. genre_list genre')
    print('6. top_rating n (where n = number of entries)')
    print('7. top_views n (where n = number of entries)')
    print('8. user_history userID')
    print('9. user_movies userID')
    print('10. user_fav_genres userID')
    print('11. user_fav_genres_multiple [userID]')
    print('12. user_comparison [userID]')
    print('13. cluster')
    print('14. recommendation model userID n (where n = number of recommendations)')


def getUserInput():
    userInput = input('Please enter your selection: ')
    return userInput


def parseUserInput(userInput):
    global useSmallDataSet
    movieTitle = re.findall(r'"([^"]*)"', userInput)
    splitInput = userInput.split()
    if userInput == "exit":
        sys.exit(1)
    elif userInput == "small":
        useSmallDataSet = True
    elif userInput == "large":
        useSmallDataSet = False
    elif splitInput[0] == "2":
        result = movies_id(splitInput[1], useSmallDataSet)
        result.show(n=2147483647, truncate=False)
    elif splitInput[0] == "3":
        result = movies_title(movieTitle[0], useSmallDataSet)
        result.show(n=2147483647, truncate=False)
    elif splitInput[0] == "4":
        result = movies_year(splitInput[1], useSmallDataSet)
        result.show(n=2147483647, truncate=False)
    elif splitInput[0] == "5":
        result = genre_movies(splitInput[1], useSmallDataSet)
        result.show(n=2147483647, truncate=False)
    elif splitInput[0] == "6":
        result = highest_rating(int(splitInput[1]), useSmallDataSet)
        result.show(n=2147483647, truncate=False)
    elif splitInput[0] == "7":
        result = highest_views(int(splitInput[1]), useSmallDataSet)
        result.show(n=2147483647, truncate=False)
    elif splitInput[0] == "8":
        result = user_history(splitInput[1], useSmallDataSet)
        result.show(n=2147483647, truncate=False)
    elif splitInput[0] == "9":
        result = movies_watched(splitInput[1], useSmallDataSet)
        result.show(n=2147483647, truncate=False)
    elif splitInput[0] == "10":
        result = genre_list(splitInput[1], useSmallDataSet)
        result.show(n=2147483647, truncate=False)
    elif splitInput[0] == "11":
        result = genre_list_multiple_users(splitInput[1], useSmallDataSet)
        result.show(n=2147483647, truncate=False)
    elif splitInput[0] == "12":
        user_comparison(splitInput[1], splitInput[2], useSmallDataSet)
    elif splitInput[0] == "13":
        cluster()
    elif splitInput[0] == "14":
        user_recommendation(splitInput[1], int(splitInput[2]), useSmallDataSet)


def start():
    while exit == False:
        printOptions()
        userInput = getUserInput()
        parseUserInput(userInput)


start()
