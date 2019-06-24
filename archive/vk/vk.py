import vk_api
import random
from time import sleep
import time


def captcha_handler(captcha):
    key = input("Enter Captcha {0}: ".format(captcha.get_url())).strip()
    return captcha.try_again(key)


def authorization():
    login = '' # your credentials
    password = ''  # your credentials
    app_id = 5607498
    api = vk_api.VkApi(login, password, app_id, captcha_handler=captcha_handler)
    api.auth()
    return api


# constants
message_for_repost = [
    'Фортуна улыбается только тем, кто к этому готов. Луи Пастер',
    'Раз в жизни Фортуна стучит в дверь каждого человека, но очень часто человек в это время сидит',
    'В ближайшей пивной и не слышит ее стука. Марк Твен',
    'Единственное, что можно с уверенностью сказать об удаче — она изменит. Мизнер',
    'Слабые люди верят в удачу, сильные — в причину и следствие. Ралф Уолдс Эмерсон',
    'Быть особо невезучим — еще один способ постоянно ощущать собственную важность.',
    'Человек, которому повезло, — это человек, который делал то, что другие только собирались делать.',
    'Любовь, как и удача, не любит, чтобы за ней гонялись.',
    'Удача — это постоянная готовность использовать случай.',
    'Сколько рождённых в рубашке ленится заработать даже на штаны!',
    'Беда не приходит одна, но и удача тоже.',
    'В удачу стоит верить,но не стоит ей доверять…',
    'Если человек не верит в удачу, у него небогатый жизненный опыт.',
    'Все плохое когда-нибудь кончается. И начинается еще худшее.',
    'Думай об успехе, представляй себе успех, и ты приведешь в действие силу, осуществляющую желания.',
    'За каждым спуском есть подъем.',
    'Последняя удача лучше первой.',
    'Первая победа — не победа.',
    'Не гляди вверх — не упадешь вниз',
    'Когда удача входит — ум выходит.',
    "Делай тихо, и тогда о тебе скажут громко. Павел Шустов"
]
search_q = [
    'подписаться репост',
    'репост подписка',
    "конкурс",
    "подписаться рассказать друзьям",
    "конкурс подписаться",
    "бесплатно репост",
    "бесплатно подписаться",
    "репост конкурс",
    "конкурс рассказать друзьям",
    "выиграть подписаться",
    "вступить репост"
]
# ban_list = open('~/PycharmProjects/small_stuff/vk/cities.txt', encoding='CP1251').read().split('\n')
ban_list = open('cities.txt', encoding='CP1251').read().split('\n')
ban_list = [x.lower() for x in ban_list]
ban_list.remove(ban_list[-1])


def reposts():
    api = authorization()
    # for q in search_q:
    # print('\n\n' + q)
    for days in range(30):
        for hours in range(24):
            search_values = {'q': random.choice(search_q),
                             'extended': 1,
                             'count': 200,
                             'start_time': time.time() - hours * 3600 - days * 3600*24,
                             'end_time': time.time() - (hours - 1) * 3600 - days * 3600*24}
            search_items = api.method('newsfeed.search', search_values)['items']
            # https://vk.com/dev/newsfeed.search look up for item details
            for item in search_items:
                try:
                    if item['reposts']['count'] > 128:
                        owner_id = item['owner_id']
                        group_name = api.method('groups.getById', {'group_id': -1 * owner_id})[0]['name'].lower()
                        # next if is needed for banning stupid groups
                        # if not any(word or city in group_name for word in ban_list for city in cities):
                        if not any(word in group_name for word in ban_list):
                            print(group_name)
                            api.method('groups.join', {'group_id': -1 * owner_id})
                            repost_v = {
                                'object': 'wall{0}_{1}'.format(owner_id, item['id']),
                                'message': random.choice(message_for_repost),
                            }
                            api.method('wall.repost', repost_v)
                            # sleep(random.randint(1, 10))
                            sleep(1728)  # it's 50 posts per day
                except vk_api.ApiError as detail:
                    print(detail)
                    continue
        print('day', days+1)


def friends():
    api = authorization()
    while True:
        api.method('friends.add', {'user_id': random.randint(50e6, 380e6), 'text': 'Привет, знаешь кто я?)'})
        sleep(random.randint(1, 10))

if __name__ == "__main__":
    while True:
        reposts()
    # friends()
