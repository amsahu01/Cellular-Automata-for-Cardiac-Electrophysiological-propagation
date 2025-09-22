from cardiac_ca import SimulationApp, SimulationConfig


def main():
    cfg = SimulationConfig()
    app = SimulationApp(cfg)
    app.run()


if __name__ == "__main__":
    main()