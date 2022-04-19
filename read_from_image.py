# done and checked

import pytesseract
from PIL import Image


def validate_id(read_id):
    """
    checks if the given is an
    Israeli ID according to a mathematical relation

    Args:
        read_id(str): the string pytesseract read \
        from the image

    Returns(bool):
    TRUE if the ID is a valid Israeli ID
    False if the ID is not a valid Israeli ID
    """
    if len(read_id) != 9:
        return False
    idList = list(map(int, read_id))
    counter = 0
    for i in range(9):
        idList[i] *= (i % 2) + 1
        if idList[i] > 9:
            idList[i] -= 9
        counter += idList[i]

    if counter % 10 == 0:
        return True
    else:
        return False


def read_from_image(path_to_cropped_image, path_to_original_image):
    """
    reads the text in the cropped image to find the ID,
    if failed, it reads the original image and should send the
    picture to the admin dashboard

    Args:
        path_to_cropped_image(str):
        path_to_original_image(str):

    Returns(str):
        returns the ID if found
        otherwise returns a failed message

    """
    img = Image.open(path_to_cropped_image)
    # enhancer = ImageEnhance.Contrast(img)
    img.show()
    text = pytesseract.image_to_string(img)
    # print('THis s the text with the link', text)
    read_string_from_image = extract_numbers(text)
    # the start and end potential ID are to contain
    # the 9 numbers under the face in a biometric ID
    start_potential_id = len(read_string_from_image) - 9
    end_potential_id = len(read_string_from_image)
    while start_potential_id >= 0:
        print(read_string_from_image[start_potential_id:end_potential_id],
              "is being checked at the moment...")
        if validate_id(read_string_from_image
                       [start_potential_id:end_potential_id]):
            print('successful', read_string_from_image,
                  'has the id :',
                  read_string_from_image[start_potential_id:end_potential_id])
            return 'has the ID', \
                   read_string_from_image[start_potential_id:end_potential_id]

        start_potential_id -= 1
        end_potential_id -= 1
    print('take another pic please')

    # if the OCR can't read the numbers under the face
    # it reads all the numbers in the image and
    # sends the image to the dashboard to validate

    img = Image.open(path_to_original_image)

    img.show()
    text = pytesseract.image_to_string(img)
    # print('THis s the text with the link', text)
    read_string_from_image = extract_numbers(text)
    start_potential_id = len(read_string_from_image) - 9
    end_potential_id = len(read_string_from_image)
    while start_potential_id > 0:
        print(read_string_from_image
              [start_potential_id:end_potential_id],
              "this is what is being checked at the moment.........")
        if validate_id(read_string_from_image
                       [start_potential_id:end_potential_id]):
            print(read_string_from_image,
                  'has the id::: please check the admin dashboard',
                  read_string_from_image[start_potential_id:end_potential_id])
            return 'has the ID'

        start_potential_id -= 1
        end_potential_id -= 1

    print('take another pic please')


def extract_numbers(word):
    """
    Takes a string, converts it to an array and removes
    everything that is not a digit
    that way we get ONLY the numbers
    Args:
        word(str): all the text read from the image

    Returns: a string containing all the numbers read in the image

    """
    arr = [char for char in word]
    print(arr)
    arr_clean = list()
    for elm in arr:
        if elm.isdigit():
            arr_clean.append(elm)
    ID = ''.join(str(x) for x in arr_clean)
    print(arr_clean, ID)
    return ID
