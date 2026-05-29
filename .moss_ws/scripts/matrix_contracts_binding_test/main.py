"""Verify all manifests.providers() contracts are bound in the IoC container."""
from ghoshell_moss.core.blueprint.matrix import Matrix


async def main():
    async with Matrix.discover() as matrix:
        providers = matrix.manifests.providers()
        if not providers:
            print("No providers found in manifests.")
            return

        total = 0
        bound = 0
        missing: list[str] = []

        for info in providers:
            contracts = [info.provider.contract()] + list(info.provider.aliases())
            for contract in contracts:
                total += 1
                name = f"{contract.__module__}.{contract.__qualname__}"
                if matrix.container.bound(contract):
                    bound += 1
                else:
                    missing.append(name)

        print(f"Providers: {len(providers)}")
        print(f"Contracts (incl. aliases): {total}")
        print(f"Bound: {bound}")
        print(f"Missing: {len(missing)}")
        if missing:
            print()
            for name in missing:
                print(f"  NOT BOUND: {name}")
            raise SystemExit(1)
        else:
            print("All contracts bound.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
