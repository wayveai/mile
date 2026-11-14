from mile.config import get_cfg, get_parser


def test_default_parser_namespace_is_accepted():
    cfg = get_cfg(get_parser().parse_args([]))
    assert cfg.TAG == 'default'


def test_config_file_aliases_and_command_line_overrides(tmp_path):
    config = tmp_path / 'experiment.yml'
    config.write_text('TAG: from_file\nBATCHSIZE: 7\n')
    for flag in ['--config', '--config-file']:
        args = get_parser().parse_args([flag, str(config), 'TAG', 'override'])
        cfg = get_cfg(args)
        assert cfg.TAG == 'override'
        assert cfg.BATCHSIZE == 7
