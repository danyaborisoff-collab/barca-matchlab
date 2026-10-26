# Источники и границы данных — v0.8

## Исходный архив

Пользователь предоставил archive.zip. Матчи: games.csv; игроки: players.csv; исторические оценки: player_valuations.csv; переходы: transfers.csv. Ссылки исходных записей ведут на Transfermarkt. Лицензия и полнота архива не подтверждены; автоматический scraping не выполнялся. Хеши оригиналов находятся в data/manifest.json.

Применён фильтр восьми турниров и стартовых лет сезонов 2012–2024. Матч — единица наблюдения, player_id — ключ игрока. Валидация дат и уникальности не заменяет внешнюю сверку каждого матча. Некоторые кубковые голы содержат пенальти. Плей-офф отделён от ML, но доступен в архивном разделе.

Рейтинги стоимости рассчитаны скриптом scripts/prepare_data.py из исторических оценок и переходов. Позиция из players.csv справочная. Срезы годовые, дата каждой оценки показана. Пропуски, старые оценки и неоднозначные переходы не считаются нулевыми ценами. Нет полного состава, матчевых показателей игроков или оценок спортивной формы.

## Подтверждённые награды

Каждая запись хранит турнир, сезон, название награды, победителя, клуб и источник. Это не автоматически вычисленный рейтинг. Источник может ретроспективно перечислять прошлых победителей. Полнота справочника не заявляется.

- https://sscnapoli.it/en/mctominay-vince-lea-sports-fc-player-of-the-year
- https://www.bundesliga.com/en/bundesliga/news/harry-kane-bundesliga-player-of-the-season-2024-25-32339
- https://www.laliga.com/en-ES/news/laliga-presents-the-2024-2025-season-awards-at-the-globe-soccer-gala-in-dubai
- https://www.premierleague.com/en/news/4314799/salah-wins-202425-ea-sports-player-of-the-season-award
- https://www.psg.fr/content/ballon-d-or-2025-une-ceremonie-historique-pour-le-paris-saint-germain-ousmane-dembele-designe-meilleur-joueur-du-monde
- https://www.uefa.com/uefachampionsleague/news/029a-1de593cb29ee-c04434a97ebb-1000--ousmane-dembele-named-2024-25-uefa-champions-league-playe/
- https://www.uefa.com/uefaconferenceleague/news/0299-1ddf890eafdc-be3746b02448-1000--isco-named-2024-25-uefa-conference-league-player-of-the-s
- https://www.uefa.com/uefaeuropaleague/news/0299-1dd17e58badd-0d32129e7f0b-1000--cristian-romero-named-2024-25-uefa-europa-league-player-of-/
