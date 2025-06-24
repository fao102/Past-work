import requests
from bs4 import BeautifulSoup
import sys


URL = "http://books.toscrape.com/"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

products = soup.find_all("article", class_="product_pod")


one_star = []
two_star = []
three_star = []
four_star = []
five_star = []


def ratingCollector(rating):

    for product in products:
        if product.find("p", class_="star-rating One") != None:
            if product not in one_star:
                one_star.append(product)

        elif product.find("p", class_="star-rating Two") != None:
            if product not in two_star:
                two_star.append(product)

        elif product.find("p", class_="star-rating Three") != None:
            if product not in three_star:
                three_star.append(product)

        elif product.find("p", class_="star-rating Four") != None:
            if product not in four_star:
                four_star.append(product)

        elif product.find("p", class_="star-rating Five") != None:
            if product not in five_star:
                five_star.append(product)

        else:
            print(":(")

    if rating == 1:
        print("////////////////////////////")
        print("One Star Rated Books:")
        for book in one_star:
            name = book.find("h3")
            print(name.text.strip())

        print("////////////////////////////")

        main()

    elif rating == 2:
        print("////////////////////////////")
        print("Two Star Rated Books:")
        for book in two_star:
            name = book.find("h3")
            print(name.text.strip())
        print("////////////////////////////")
        main()

    elif rating == 3:
        print("////////////////////////////")
        print("Three Star Rated Books:")
        for book in three_star:
            name = book.find("h3")
            print(name.text.strip())
        print("////////////////////////////")
        main()

    elif rating == 4:
        print("////////////////////////////")
        print("Four Star Rated Books:")
        for book in four_star:
            name = book.find("h3")
            print(name.text.strip())
        print("////////////////////////////")
        main()

    elif rating == 5:
        print("////////////////////////////")
        print("Five Star Rated Books:")
        for book in five_star:
            name = book.find("h3")
            print(name.text.strip())
        print("////////////////////////////")
        main()

    elif rating == 8:
        genres(soup)

    elif rating == 9:
        price()

    else:
        print("Try again, incorrect input")
        main()


def price():
    total_price = 0
    num = len(products)
    for product in products:
        price = product.find("p", class_="price_color")
        total_price += int(float(price.text.strip("£")))

    avg_price = total_price / num

    print(f"the total price of every book on this website is £{total_price}")
    print(f"the average price of a book on this website is £{avg_price}")
    main()


def names():
    for product in products:
        name = product.find("h3")
        print(name.text.strip())


def genres(data):
    genres = soup.find_all("ul", class_="nav nav-list")
    for genre in genres:
        name = genre.find("li")
        print(name.text.strip())
    main()


def main():
    rating = int(
        input(
            "Enter a rating or type 0 to exit, type 9 for prices and type 8 for genres "
        )
    )
    if rating != 0 or rating != 8 or rating != 9:
        ratingCollector(rating)

    elif rating == 0:
        sys.exit()

    # if
    # names()
    # price()
    # genres(soup)


main()
