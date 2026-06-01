from click.testing import CliRunner

from fluctuant.cli import main


def test_main_help():
    result = CliRunner().invoke(main, ['--help'])
    assert result.exit_code == 0
    assert 'astromer' in result.output
    assert 'physics' in result.output


def test_astromer_help():
    result = CliRunner().invoke(main, ['astromer', '--help'])
    assert result.exit_code == 0
    for flag in ('--n-agn', '--n-tde', '--seed', '--weights', '--tune', '--augment', '--output-dir'):
        assert flag in result.output


def test_physics_help():
    result = CliRunner().invoke(main, ['physics', '--help'])
    assert result.exit_code == 0
    for flag in ('--n-agn', '--n-tde', '--seed', '--output-dir'):
        assert flag in result.output


def test_unknown_command_exits_nonzero():
    result = CliRunner().invoke(main, ['nonexistent'])
    assert result.exit_code != 0
