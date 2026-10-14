import main as demo


def test_demo_runs_end_to_end():
    # main() asserts its own sanity checks: tree shape, numbering, verdicts, hidden tests
    demo.main()
