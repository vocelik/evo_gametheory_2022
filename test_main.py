import axelrod as axl

import main


def test_set_PLAYER_Heterogeneity():
    players = axl.Cooperator(), axl.Defector()
    masses = 1.5, 2
    independences = 3, 2
    main.set_PLAYER_heterogeneity(
        PLAYERS=players, masses=masses, independences=independences
    )
    for player, mass, independence, id_ in zip(
        players, masses, independences, range(1, 3)
    ):
        assert player.mass == mass
        assert player.independence == independence
        assert player.id == id_
