from numbers_extraction import id_numbers_locate


def chose_method(path, chosen_method):
    coordinates = []
    if chosen_method == 'Biometric_ID':
        coordinates = [1, 1.5, 0.9, 1.1]
    if chosen_method == 'Driving_License':
        coordinates = [1, 1.5, 0.9, 1.1]
    if chosen_method == 'Non_Biometric_ID':
        coordinates = [1, 1.5, 0.9, 1.1]
    if chosen_method == 'Passports':
        coordinates = [1, 1.5, 0.9, 1.1]
    if chosen_method == 'Visot':
        coordinates = [1, 1.5, 0.9, 1.1]
    return id_numbers_locate(path, coordinates)

