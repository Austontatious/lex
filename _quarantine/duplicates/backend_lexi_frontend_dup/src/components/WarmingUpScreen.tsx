import React from "react";
import { Flex, Spinner, Text } from "@chakra-ui/react";

const WarmingUpScreen: React.FC = () => {
  return (
    <Flex
      direction="column"
      align="center"
      justify="center"
      height="100vh"
      bg="gray.800"
      color="white"
    >
      <Spinner size="xl" thickness="4px" speed="0.65s" color="blue.400" mb={4} />
      <Text fontSize="xl">FRIDAY is warming up... Loading tensors, memory, and charm 💫</Text>
    </Flex>
  );
};

export default WarmingUpScreen;

