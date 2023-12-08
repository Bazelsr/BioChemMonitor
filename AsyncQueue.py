class AsyncQueue:

    def __init__(self):
        self.__stack = []

    async def insert(self,val):
        self.__stack.append(val)

    async def pop(self):
        return self.__stack.pop()

    async def get_size(self):
        return len(self.__stack)

    async def as_list(self):
        return list(self.__stack)

    async def is_empty(self):
        return (await self.get_size() == 0)

